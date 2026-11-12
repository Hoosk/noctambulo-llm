//! Preallocated ring KV caches, heavy-hitter eviction, and host-RAM spill/replay.

use std::sync::atomic::{AtomicU64, Ordering};

use burn::tensor::{Bool, DType, Device, Int, Tensor, TensorData};

use super::quant::{
    KvCacheDtype, assert_f16_storage, cache_storage_dtype, dequantize_rows, quant_block,
    quantize_rows,
};

/// Preallocated per-layer KV cache.
///
/// The buffers are sized once (capacity = trained context window, possibly
/// extended by RoPE scaling) and written with `slice_assign`, which mutates in
/// place when the handle has a single owner. This removes the per-token
/// `Tensor::cat` reallocation the engine used to do, which copied the whole
/// cache on every decoded token. Quantized dtypes store packed i32 words plus
/// One evicted cache row archived in host RAM, in f32 and flat
/// (head, dim) layout for K/V (or the latent for MLA).
#[derive(Clone, Debug)]
pub(crate) struct SpilledRow {
    /// Absolute position the row was cached at.
    pub(crate) position: usize,
    /// Accumulated attention score when the row was evicted.
    pub(crate) score: f32,
    pub(crate) k: Vec<f32>,
    /// Empty for MLA caches (only the latent is stored).
    pub(crate) v: Vec<f32>,
}

/// Host-RAM archive of rows evicted by heavy-hitter eviction. Rows are stored
/// in f32 regardless of the cache storage dtype, so a replayed row re-enters
/// the cache through the same quantization path as a fresh row. Bounded by the
/// model context window (`max_seq` rows at most).
#[derive(Clone, Debug, Default)]
pub struct SpillStore {
    pub(crate) rows: Vec<SpilledRow>,
}

impl SpillStore {
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    /// Host bytes held by the archive (values + position + score + lengths).
    pub fn bytes(&self) -> usize {
        self.rows
            .iter()
            .map(|r| 2 * std::mem::size_of::<usize>() + 4 * (r.k.len() + r.v.len() + 1))
            .sum()
    }

    fn push(&mut self, row: SpilledRow) {
        self.rows.push(row);
    }

    /// Highest-score row first; ties break towards the newest position, which
    /// is what a denser cache would have kept.
    fn pop_best(&mut self) -> Option<SpilledRow> {
        let idx = self
            .rows
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                a.score
                    .partial_cmp(&b.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(a.position.cmp(&b.position))
            })
            .map(|(i, _)| i)?;
        Some(self.rows.swap_remove(idx))
    }
}

/// per-block f32 scales.
#[derive(Clone, Debug)]
pub struct KvCache {
    pub k: Option<Tensor<4>>,
    pub v: Option<Tensor<4>>,
    /// Packed quantized K rows (or the MLA latent) `[capacity, words]`.
    pub kq: Option<Tensor<2, Int>>,
    pub ks: Option<Tensor<2>>,
    pub vq: Option<Tensor<2, Int>>,
    pub vs: Option<Tensor<2>>,
    /// Number of valid rows (`<= budget`).
    pub len: usize,
    /// Physical rows allocated (the replay ceiling when the budget grows).
    pub capacity: usize,
    /// Absolute position stored in each ring slot (`usize::MAX` = empty).
    pub positions: Vec<usize>,
    /// Storage format for the cached rows.
    pub dtype: KvCacheDtype,
    /// Accumulated attention mass per slot, always f32. `Some` enables
    /// heavy-hitter eviction; the accumulator is independent of the storage
    /// dtype (f16/int8/q4 caches still score in f32).
    pub scores: Option<Tensor<1>>,
    /// Most recent tokens protected from eviction (0 when disabled).
    pub recent: usize,
    /// Current resident limit (`<= capacity`); grows with spill replay.
    pub budget: usize,
    /// Host-RAM archive of evicted rows (`None` = eviction discards them).
    pub spill: Option<SpillStore>,
    /// Rows archived by eviction (cumulative, for reporting).
    pub spilled_total: usize,
    /// Rows restored by replay (cumulative, for reporting).
    pub replayed_total: usize,
}

impl KvCache {
    /// Allocate zeroed buffers up front, sized to `capacity` positions.
    pub fn preallocated(
        batch: usize,
        n_kv_head: usize,
        head_dim: usize,
        capacity: usize,
        dtype: KvCacheDtype,
        device: &Device,
    ) -> Self {
        if dtype == KvCacheDtype::F16 {
            assert_f16_storage(device);
        }
        if dtype.is_quantized() {
            let row_len = n_kv_head * head_dim;
            let pack = dtype.pack();
            let block = quant_block(row_len);
            assert!(
                row_len.is_multiple_of(pack),
                "KV row length {row_len} must be divisible by {pack} for {} packing",
                dtype.as_str()
            );
            assert!(
                row_len.is_multiple_of(block),
                "KV row length {row_len} must be divisible by {block} for block scales"
            );
            let data: Tensor<2, Int> = Tensor::zeros([capacity, row_len / pack], device);
            let scales: Tensor<2> = Tensor::ones([capacity, row_len / block], (device, DType::F32));
            return Self {
                k: None,
                v: None,
                kq: Some(data.clone()),
                ks: Some(scales.clone()),
                vq: Some(data),
                vs: Some(scales),
                len: 0,
                capacity,
                positions: vec![usize::MAX; capacity],
                dtype,
                scores: None,
                recent: 0,
                budget: capacity,
                spill: None,
                spilled_total: 0,
                replayed_total: 0,
            };
        }
        let shape = [batch, n_kv_head, capacity, head_dim];
        let storage = cache_storage_dtype(dtype);
        let k: Tensor<4> = Tensor::zeros(shape, (device, storage));
        let v: Tensor<4> = Tensor::zeros(shape, (device, storage));
        Self {
            k: Some(k),
            v: Some(v),
            kq: None,
            ks: None,
            vq: None,
            vs: None,
            len: 0,
            capacity,
            positions: vec![usize::MAX; capacity],
            dtype,
            scores: None,
            recent: 0,
            budget: capacity,
            spill: None,
            spilled_total: 0,
            replayed_total: 0,
        }
    }

    /// Ring slot for an absolute position. The first `sink_tokens` slots are
    /// reserved for the attention sinks and never overwritten; the rest ring.
    pub fn slot_for(&self, pos: usize, sink_tokens: usize) -> usize {
        if pos < sink_tokens {
            pos
        } else {
            let ring = self.capacity - sink_tokens;
            sink_tokens + (pos - sink_tokens) % ring
        }
    }

    /// MLA variant: only the latent buffer (`k`) is allocated; `v` stays
    /// unused because K and V are expanded from the same latent.
    pub fn preallocated_single(
        batch: usize,
        heads: usize,
        dim: usize,
        capacity: usize,
        dtype: KvCacheDtype,
        device: &Device,
    ) -> Self {
        if dtype == KvCacheDtype::F16 {
            assert_f16_storage(device);
        }
        if dtype.is_quantized() {
            let row_len = heads * dim;
            let pack = dtype.pack();
            let block = quant_block(row_len);
            assert!(
                row_len.is_multiple_of(pack),
                "MLA latent row {row_len} must be divisible by {pack} for {} packing",
                dtype.as_str()
            );
            assert!(
                row_len.is_multiple_of(block),
                "MLA latent row {row_len} must be divisible by {block} for block scales"
            );
            let data: Tensor<2, Int> = Tensor::zeros([capacity, row_len / pack], device);
            let scales: Tensor<2> = Tensor::ones([capacity, row_len / block], (device, DType::F32));
            return Self {
                k: None,
                v: None,
                kq: Some(data),
                ks: Some(scales),
                vq: None,
                vs: None,
                len: 0,
                capacity,
                positions: vec![usize::MAX; capacity],
                dtype,
                scores: None,
                recent: 0,
                budget: capacity,
                spill: None,
                spilled_total: 0,
                replayed_total: 0,
            };
        }
        let k: Tensor<4> = Tensor::zeros(
            [batch, heads, capacity, dim],
            (device, cache_storage_dtype(dtype)),
        );
        Self {
            k: Some(k),
            v: None,
            kq: None,
            ks: None,
            vq: None,
            vs: None,
            len: 0,
            capacity,
            positions: vec![usize::MAX; capacity],
            dtype,
            scores: None,
            recent: 0,
            budget: capacity,
            spill: None,
            spilled_total: 0,
            replayed_total: 0,
        }
    }

    /// Full version of the eviction setup: bounds the resident rows to
    /// `budget` (`<= capacity`) and, when `spill` is set, archives evicted rows
    /// in host RAM so they can be replayed into slots freed by budget growth.
    pub fn enable_eviction_with(&mut self, recent: usize, budget: usize, spill: bool) {
        let device = match (&self.k, &self.v, &self.kq) {
            (Some(k), _, _) => k.device(),
            (_, Some(v), _) => v.device(),
            (_, _, Some(q)) => q.device(),
            _ => panic!("KV cache has no storage to derive the device from"),
        };
        self.scores = Some(Tensor::zeros([self.capacity], &device));
        self.recent = recent;
        self.budget = budget.clamp(1, self.capacity);
        self.spill = spill.then(SpillStore::default);
        self.spilled_total = 0;
        self.replayed_total = 0;
    }

    pub fn eviction_enabled(&self) -> bool {
        self.scores.is_some()
    }

    /// Grows the resident budget by `rows` (clamped to `capacity`) and replays
    /// the highest-score archived rows to fill the new slots. Returns the
    /// number of replayed rows.
    pub fn grow_and_replay(&mut self, rows: usize) -> usize {
        self.budget = (self.budget + rows).min(self.capacity);
        self.replay_spilled()
    }

    /// Replays the highest-score archived rows until the resident budget is
    /// full or the archive runs out. Returns the number of replayed rows.
    pub fn replay_spilled(&mut self) -> usize {
        if self.spill.is_none() {
            return 0;
        }
        let mut rows = Vec::new();
        while self.len < self.budget {
            match self.spill.as_mut().and_then(SpillStore::pop_best) {
                Some(row) => rows.push(row),
                None => break,
            }
        }
        let replayed = rows.len();
        for row in rows {
            self.write_replayed_row(row);
        }
        self.replayed_total += replayed;
        replayed
    }

    /// Writes an archived row into the first free slot (`len`, which is always
    /// virgin after a budget growth) restoring its position and score. The row
    /// re-enters through the storage quantization path, like a fresh row.
    fn write_replayed_row(&mut self, row: SpilledRow) {
        let slot = self.len;
        assert!(slot < self.capacity, "replay needs a free cache slot");

        if self.dtype.is_quantized() {
            let words = self
                .kq
                .as_ref()
                .expect("quantized cache kq buffer")
                .dims()[1];
            let row_len = words * self.dtype.pack();
            let blocks = row_len / quant_block(row_len);
            let device = self
                .kq
                .as_ref()
                .expect("quantized cache kq buffer")
                .device();
            let k = Tensor::<2>::from_data(TensorData::new(row.k, [1, row_len]), &device);
            let (kq_row, ks_row) = quantize_rows(k, self.dtype);
            let mut kq = self.kq.take().expect("kv cache kq buffer");
            kq = kq.slice_assign([slot..slot + 1, 0..words], kq_row);
            self.kq = Some(kq);
            let mut ks = self.ks.take().expect("kv cache ks buffer");
            ks = ks.slice_assign([slot..slot + 1, 0..blocks], ks_row);
            self.ks = Some(ks);

            if !row.v.is_empty() {
                let v = Tensor::<2>::from_data(TensorData::new(row.v, [1, row_len]), &device);
                let (vq_row, vs_row) = quantize_rows(v, self.dtype);
                let mut vq = self.vq.take().expect("kv cache vq buffer");
                vq = vq.slice_assign([slot..slot + 1, 0..words], vq_row);
                self.vq = Some(vq);
                let mut vs = self.vs.take().expect("kv cache vs buffer");
                vs = vs.slice_assign([slot..slot + 1, 0..blocks], vs_row);
                self.vs = Some(vs);
            }
        } else {
            let (heads, dim, device) = match (&self.k, &self.v) {
                (Some(k), _) => {
                    let dims = k.dims();
                    (dims[1], dims[3], k.device())
                }
                (_, Some(v)) => {
                    let dims = v.dims();
                    (dims[1], dims[3], v.device())
                }
                _ => panic!("kv cache has no storage"),
            };
            let to_storage = |t: Tensor<4>| {
                t.cast(cache_storage_dtype(self.dtype))
            };
            let k = Tensor::<4>::from_data(
                TensorData::new(row.k, [1, heads, 1, dim]),
                &device,
            );
            let mut k_buf = self.k.take().expect("kv cache k buffer");
            k_buf = k_buf.slice_assign([0..1, 0..heads, slot..slot + 1, 0..dim], to_storage(k));
            self.k = Some(k_buf);

            if !row.v.is_empty() {
                let v = Tensor::<4>::from_data(
                    TensorData::new(row.v, [1, heads, 1, dim]),
                    &device,
                );
                let mut v_buf = self.v.take().expect("kv cache v buffer");
                v_buf = v_buf.slice_assign([0..1, 0..heads, slot..slot + 1, 0..dim], to_storage(v));
                self.v = Some(v_buf);
            }
        }

        self.positions[slot] = row.position;
        let scores = self.scores.take().expect("eviction scores accumulator");
        let device = scores.device();
        let value = Tensor::<1>::from_data(TensorData::new(vec![row.score], [1]), &device);
        // `slice_assign` takes `[Range; N]`; the single-range array is the
        // intended API here.
        #[allow(clippy::single_range_in_vec_init)]
        let scores = scores.slice_assign([slot..slot + 1], value);
        self.scores = Some(scores);
        self.len += 1;
    }
}

/// Contiguous ring segments `(slot, offset, len)` covering the `new_seq_len`
/// rows written at `start_pos`.
fn cache_segments(
    cache: &KvCache,
    start_pos: usize,
    new_seq_len: usize,
    sink_tokens: usize,
) -> Vec<(usize, usize, usize)> {
    let mut segments = Vec::new();
    let mut offset = 0usize;
    while offset < new_seq_len {
        let pos = start_pos + offset;
        let slot = cache.slot_for(pos, sink_tokens);
        let mut len = 1usize;
        while offset + len < new_seq_len {
            let next_pos = start_pos + offset + len;
            if cache.slot_for(next_pos, sink_tokens) == slot + len {
                len += 1;
            } else {
                break;
            }
        }
        segments.push((slot, offset, len));
        offset += len;
    }
    segments
}

/// Training-time eviction simulation step counter: every forward pass samples
/// a different heavy-hitter keep set, so the model cannot overfit one cache
/// layout.
static KEEP_STEP: AtomicU64 = AtomicU64::new(0);

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// Column keep-mask that simulates the steady-state bounded cache: sinks and
/// recent columns are handled positionally by the mask, while old columns
/// survive with probability `hitters / (n_ctx - sinks - recent)`.
pub(crate) fn sample_keep_mask(
    seq_len: usize,
    n_ctx: usize,
    budget: usize,
    recent: usize,
    sinks: usize,
    device: &Device,
) -> Tensor<2, Bool> {
    let old_region = n_ctx.saturating_sub(sinks + recent).max(1);
    let hitters = budget.saturating_sub(sinks + recent);
    let p = (hitters as f64 / old_region as f64).min(1.0);
    let step = KEEP_STEP.fetch_add(1, Ordering::Relaxed);
    let mut bits = vec![false; seq_len];
    for (c, bit) in bits.iter_mut().enumerate() {
        let h = splitmix64(step ^ (c as u64).wrapping_mul(0x9E3779B97F4A7C15));
        let u = (h >> 11) as f64 / (1u64 << 53) as f64;
        *bit = u < p;
    }
    Tensor::<1, Bool>::from_data(TensorData::new(bits, [seq_len]), device)
        .reshape([1, seq_len])
}

/// Slot assigned to every new row when heavy-hitter eviction is enabled.
/// While the cache has free rows they are appended in order; once full the
/// victim is the live row with the lowest accumulated attention score that is
/// neither a sink, nor inside the recent window, nor written by this same
/// call. `positions` and `len` are updated here; victim scores reset to zero.
pub(crate) fn alloc_evict_slots(
    cache: &mut KvCache,
    start_pos: usize,
    new_seq_len: usize,
    sink_tokens: usize,
) -> Vec<usize> {
    assert!(
        cache.budget > sink_tokens + cache.recent,
        "KV budget ({}) must exceed sink tokens ({}) + recent tokens ({})",
        cache.budget,
        sink_tokens,
        cache.recent
    );
    let scores_host: Option<Vec<f32>> = if cache.len + new_seq_len > cache.budget {
        Some(
            cache
                .scores
                .as_ref()
                .expect("eviction allocator needs the score accumulator")
                .clone()
                .into_data()
                .try_to_vec::<f32>()
                .expect("f32 score readback"),
        )
    } else {
        None
    };

    let mut slots = Vec::with_capacity(new_seq_len);
    let mut victims: Vec<usize> = Vec::new();
    // Slots assigned by this same call but not written yet (their row data
    // only exists after `alloc` returns), so they cannot be archived.
    let mut appended: Vec<usize> = Vec::new();
    // (slot, old position, old score) of every recycled row, archived before
    // the buffer is overwritten.
    let mut archived: Vec<(usize, usize, f32)> = Vec::new();
    for offset in 0..new_seq_len {
        let pos = start_pos + offset;
        if cache.len < cache.budget {
            let slot = cache.len;
            cache.len += 1;
            cache.positions[slot] = pos;
            appended.push(slot);
            slots.push(slot);
            continue;
        }
        let scores = scores_host.as_ref().expect("scores read back when full");
        // Rows at or after `cutoff` are the recent window and stay put.
        let cutoff = pos.saturating_sub(cache.recent);
        let victim = (0..cache.capacity)
            .filter(|&s| {
                let p = cache.positions[s];
                p != usize::MAX
                    && p >= sink_tokens
                    && p < cutoff
                    && !victims.contains(&s)
            })
            .min_by(|&a, &b| {
                scores[a]
                    .partial_cmp(&scores[b])
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| cache.positions[a].cmp(&cache.positions[b]))
            })
            .expect("budget must exceed sinks + recent so a victim exists");
        if appended.contains(&victim) {
            // Reusing a slot assigned earlier in this call: its row was never
            // written, so there is nothing to archive. The resident count is
            // unchanged (one row replaces another).
        } else {
            archived.push((victim, cache.positions[victim], scores[victim]));
        }
        cache.positions[victim] = pos;
        victims.push(victim);
        slots.push(victim);
    }

    if !archived.is_empty() && cache.spill.is_some() {
        spill_rows(cache, &archived);
    }

    if !victims.is_empty() {
        // Fresh rows start from a clean score; multiply by a 0/1 mask (Burn's
        // scatter only implements `Add` and its Int dtype plumbing differs per
        // backend, so the mask is the portable option).
        let mut scores = cache.scores.take().expect("scores enabled");
        let device = scores.device();
        let mask: Vec<f32> = (0..cache.capacity)
            .map(|s| if victims.contains(&s) { 0.0 } else { 1.0 })
            .collect();
        let mask = Tensor::<1>::from_data(
            TensorData::new(mask, [cache.capacity]),
            &device,
        );
        scores = scores * mask;
        cache.scores = Some(scores);
    }
    slots
}

/// Archives the rows at `(slot, old_position, old_score)` in the cache spill
/// store as f32 flat vectors. Reads every victim with one device transfer by
/// concatenating their slices first.
fn spill_rows(cache: &mut KvCache, rows: &[(usize, usize, f32)]) {
    let slots: Vec<usize> = rows.iter().map(|r| r.0).collect();
    let (k_rows, v_rows) = read_rows_f32(cache, &slots);
    let n = slots.len();
    let row_len = k_rows.len() / n;
    let spill = cache.spill.as_mut().expect("spill store enabled");
    for (i, (_, position, score)) in rows.iter().enumerate() {
        let k = k_rows[i * row_len..(i + 1) * row_len].to_vec();
        let v = if v_rows.is_empty() {
            Vec::new()
        } else {
            v_rows[i * row_len..(i + 1) * row_len].to_vec()
        };
        spill.push(SpilledRow {
            position: *position,
            score: *score,
            k,
            v,
        });
    }
    cache.spilled_total += n;
}

/// Reads the cache rows at `slots` as f32 flat `(head, dim)` vectors (the
/// latent for MLA caches). `v` is empty when the cache stores no V buffer.
fn read_rows_f32(cache: &KvCache, slots: &[usize]) -> (Vec<f32>, Vec<f32>) {
    if cache.dtype.is_quantized() {
        let kq = cache.kq.as_ref().expect("quantized cache kq buffer");
        let words = kq.dims()[1];
        let k_parts: Vec<_> = slots
            .iter()
            .map(|&s| kq.clone().slice([s..s + 1, 0..words]))
            .collect();
        let k_packed = Tensor::cat(k_parts, 0);
        let ks = cache.ks.as_ref().expect("quantized cache ks buffer");
        let blocks = ks.dims()[1];
        let ks_parts: Vec<_> = slots
            .iter()
            .map(|&s| ks.clone().slice([s..s + 1, 0..blocks]))
            .collect();
        let k_scales = Tensor::cat(ks_parts, 0);
        let k = dequantize_rows(k_packed, k_scales, cache.dtype)
            .into_data()
            .convert::<f32>()
            .try_to_vec::<f32>()
            .expect("f32 dequantized rows");

        let mut v = Vec::new();
        if let Some(vq) = &cache.vq {
            let v_parts: Vec<_> = slots
                .iter()
                .map(|&s| vq.clone().slice([s..s + 1, 0..words]))
                .collect();
            let v_packed = Tensor::cat(v_parts, 0);
            let vs = cache.vs.as_ref().expect("quantized cache vs buffer");
            let vs_parts: Vec<_> = slots
                .iter()
                .map(|&s| vs.clone().slice([s..s + 1, 0..blocks]))
                .collect();
            let v_scales = Tensor::cat(vs_parts, 0);
            v = dequantize_rows(v_packed, v_scales, cache.dtype)
                .into_data()
                .convert::<f32>()
                .try_to_vec::<f32>()
                .expect("f32 dequantized rows");
        }
        return (k, v);
    }

    let read = |buf: &Tensor<4>| -> Vec<f32> {
        let parts: Vec<_> = slots
            .iter()
            .map(|&s| buf.clone().slice([0..1, 0..buf.dims()[1], s..s + 1, 0..buf.dims()[3]]))
            .collect();
        // [1, slots, heads, dim] so the flat layout of each row is head-major.
        let joined = Tensor::cat(parts, 2).swap_dims(1, 2);
        joined
            .into_data()
            .convert::<f32>()
            .try_to_vec::<f32>()
            .expect("f32 cache rows")
    };
    let k = read(cache.k.as_ref().expect("kv cache k buffer"));
    let v = match &cache.v {
        Some(v_buf) => read(v_buf),
        None => Vec::new(),
    };
    (k, v)
}

/// Writes K/V rows for the absolute positions `start_pos..` into the ring
/// cache with contiguous `slice_assign` segments (in place when the buffer has
/// a single owner). Shared by every group leader.
pub(crate) fn cache_write(
    cache: &mut KvCache,
    k_embed: Tensor<4>,
    v: Tensor<4>,
    start_pos: usize,
    sink_tokens: usize,
) {
    let [batch_size, n_kv_head, new_seq_len, head_size] = k_embed.dims();
    assert!(
        cache.capacity > sink_tokens,
        "KV cache capacity ({}) must exceed the sink tokens ({})",
        cache.capacity,
        sink_tokens
    );
    if cache.eviction_enabled() {
        cache_write_evict(cache, k_embed, v, start_pos, sink_tokens);
        return;
    }
    for pos in start_pos..(start_pos + new_seq_len) {
        let slot = cache.slot_for(pos, sink_tokens);
        cache.positions[slot] = pos;
    }
    let segments = cache_segments(cache, start_pos, new_seq_len, sink_tokens);

    if cache.dtype.is_quantized() {
        let row_len = n_kv_head * head_size;
        let pack = cache.dtype.pack();
        let block = quant_block(row_len);
        let words = row_len / pack;
        let blocks = row_len / block;
        let to_rows =
            |x: Tensor<4>| x.swap_dims(1, 2).reshape([batch_size * new_seq_len, row_len]);
        let (kp, ksc) = quantize_rows(to_rows(k_embed), cache.dtype);
        let (vp, vsc) = quantize_rows(to_rows(v), cache.dtype);
        let mut kq = cache.kq.take().expect("kv cache kq buffer");
        let mut ks = cache.ks.take().expect("kv cache ks buffer");
        let mut vq = cache.vq.take().expect("kv cache vq buffer");
        let mut vs = cache.vs.take().expect("kv cache vs buffer");
        for (slot, offset, len) in segments {
            kq = kq.slice_assign(
                [slot..slot + len, 0..words],
                kp.clone().slice([offset..offset + len, 0..words]),
            );
            ks = ks.slice_assign(
                [slot..slot + len, 0..blocks],
                ksc.clone().slice([offset..offset + len, 0..blocks]),
            );
            vq = vq.slice_assign(
                [slot..slot + len, 0..words],
                vp.clone().slice([offset..offset + len, 0..words]),
            );
            vs = vs.slice_assign(
                [slot..slot + len, 0..blocks],
                vsc.clone().slice([offset..offset + len, 0..blocks]),
            );
        }
        cache.len = (cache.len + new_seq_len).min(cache.capacity);
        cache.kq = Some(kq);
        cache.ks = Some(ks);
        cache.vq = Some(vq);
        cache.vs = Some(vs);
        return;
    }

    let storage = cache_storage_dtype(cache.dtype);
    let k_embed = if k_embed.dtype() != storage {
        k_embed.cast(storage)
    } else {
        k_embed
    };
    let v = if v.dtype() != storage { v.cast(storage) } else { v };
    let mut k_buf = cache.k.take().expect("kv cache k buffer");
    let mut v_buf = cache.v.take().expect("kv cache v buffer");
    for (slot, offset, len) in segments {
        let write_slice = [0..batch_size, 0..n_kv_head, slot..(slot + len), 0..head_size];
        let k_part = k_embed
            .clone()
            .slice([0..batch_size, 0..n_kv_head, offset..(offset + len), 0..head_size]);
        let v_part = v
            .clone()
            .slice([0..batch_size, 0..n_kv_head, offset..(offset + len), 0..head_size]);
        k_buf = k_buf.slice_assign(write_slice.clone(), k_part);
        v_buf = v_buf.slice_assign(write_slice, v_part);
    }
    cache.len = (cache.len + new_seq_len).min(cache.capacity);
    cache.k = Some(k_buf);
    cache.v = Some(v_buf);
}

/// Heavy-hitter variant of [`cache_write`]: rows go to the slots chosen by
/// [`alloc_evict_slots`] (append while there is room, otherwise the victim
/// slot), written one row at a time because victims are not contiguous.
fn cache_write_evict(
    cache: &mut KvCache,
    k_embed: Tensor<4>,
    v: Tensor<4>,
    start_pos: usize,
    sink_tokens: usize,
) {
    let [batch_size, n_kv_head, new_seq_len, head_size] = k_embed.dims();
    let slots = alloc_evict_slots(cache, start_pos, new_seq_len, sink_tokens);

    if cache.dtype.is_quantized() {
        let row_len = n_kv_head * head_size;
        let pack = cache.dtype.pack();
        let block = quant_block(row_len);
        let words = row_len / pack;
        let blocks = row_len / block;
        let to_rows =
            |x: Tensor<4>| x.swap_dims(1, 2).reshape([batch_size * new_seq_len, row_len]);
        let (kp, ksc) = quantize_rows(to_rows(k_embed), cache.dtype);
        let (vp, vsc) = quantize_rows(to_rows(v), cache.dtype);
        let mut kq = cache.kq.take().expect("kv cache kq buffer");
        let mut ks = cache.ks.take().expect("kv cache ks buffer");
        let mut vq = cache.vq.take().expect("kv cache vq buffer");
        let mut vs = cache.vs.take().expect("kv cache vs buffer");
        for (row, &slot) in slots.iter().enumerate() {
            kq = kq.slice_assign(
                [slot..slot + 1, 0..words],
                kp.clone().slice([row..row + 1, 0..words]),
            );
            ks = ks.slice_assign(
                [slot..slot + 1, 0..blocks],
                ksc.clone().slice([row..row + 1, 0..blocks]),
            );
            vq = vq.slice_assign(
                [slot..slot + 1, 0..words],
                vp.clone().slice([row..row + 1, 0..words]),
            );
            vs = vs.slice_assign(
                [slot..slot + 1, 0..blocks],
                vsc.clone().slice([row..row + 1, 0..blocks]),
            );
        }
        cache.kq = Some(kq);
        cache.ks = Some(ks);
        cache.vq = Some(vq);
        cache.vs = Some(vs);
        return;
    }

    let storage = cache_storage_dtype(cache.dtype);
    let k_embed = if k_embed.dtype() != storage {
        k_embed.cast(storage)
    } else {
        k_embed
    };
    let v = if v.dtype() != storage { v.cast(storage) } else { v };
    let mut k_buf = cache.k.take().expect("kv cache k buffer");
    let mut v_buf = cache.v.take().expect("kv cache v buffer");
    for (row, &slot) in slots.iter().enumerate() {
        let write_slice = [0..batch_size, 0..n_kv_head, slot..slot + 1, 0..head_size];
        let k_part = k_embed
            .clone()
            .slice([0..batch_size, 0..n_kv_head, row..row + 1, 0..head_size]);
        let v_part = v
            .clone()
            .slice([0..batch_size, 0..n_kv_head, row..row + 1, 0..head_size]);
        k_buf = k_buf.slice_assign(write_slice.clone(), k_part);
        v_buf = v_buf.slice_assign(write_slice, v_part);
    }
    cache.k = Some(k_buf);
    cache.v = Some(v_buf);
}

/// Absolute position of every ring slot as `i32` (`i32::MAX` for empty slots,
/// masked out by the attention).
pub(crate) fn cache_positions_i32(cache: &KvCache) -> Vec<i32> {
    cache
        .positions
        .iter()
        .map(|&p| if p == usize::MAX { i32::MAX } else { p as i32 })
        .collect()
}

/// MLA variant of [`cache_write`]: only the latent (`k`) buffer is written.
pub(crate) fn cache_write_latent(
    cache: &mut KvCache,
    latent: Tensor<4>,
    start_pos: usize,
    sink_tokens: usize,
) {
    let [batch_size, heads, new_seq_len, dim] = latent.dims();
    assert!(
        cache.capacity > sink_tokens,
        "KV cache capacity ({}) must exceed the sink tokens ({})",
        cache.capacity,
        sink_tokens
    );
    if cache.eviction_enabled() {
        cache_write_latent_evict(cache, latent, start_pos, sink_tokens);
        return;
    }
    for pos in start_pos..(start_pos + new_seq_len) {
        let slot = cache.slot_for(pos, sink_tokens);
        cache.positions[slot] = pos;
    }
    let segments = cache_segments(cache, start_pos, new_seq_len, sink_tokens);

    if cache.dtype.is_quantized() {
        let row_len = heads * dim;
        let pack = cache.dtype.pack();
        let block = quant_block(row_len);
        let words = row_len / pack;
        let blocks = row_len / block;
        let rows = latent
            .swap_dims(1, 2)
            .reshape([batch_size * new_seq_len, row_len]);
        let (packed, scales) = quantize_rows(rows, cache.dtype);
        let mut kq = cache.kq.take().expect("mla cache kq buffer");
        let mut ks = cache.ks.take().expect("mla cache ks buffer");
        for (slot, offset, len) in segments {
            kq = kq.slice_assign(
                [slot..slot + len, 0..words],
                packed.clone().slice([offset..offset + len, 0..words]),
            );
            ks = ks.slice_assign(
                [slot..slot + len, 0..blocks],
                scales.clone().slice([offset..offset + len, 0..blocks]),
            );
        }
        cache.len = (cache.len + new_seq_len).min(cache.capacity);
        cache.kq = Some(kq);
        cache.ks = Some(ks);
        return;
    }

    let storage = cache_storage_dtype(cache.dtype);
    let latent = if latent.dtype() != storage {
        latent.cast(storage)
    } else {
        latent
    };
    let mut k_buf = cache.k.take().expect("kv cache latent buffer");
    for (slot, offset, len) in segments {
        let write_slice = [0..batch_size, 0..heads, slot..(slot + len), 0..dim];
        let part = latent
            .clone()
            .slice([0..batch_size, 0..heads, offset..(offset + len), 0..dim]);
        k_buf = k_buf.slice_assign(write_slice, part);
    }
    cache.len = (cache.len + new_seq_len).min(cache.capacity);
    cache.k = Some(k_buf);
}

/// Heavy-hitter variant of [`cache_write_latent`], one row at a time into the
/// slots chosen by [`alloc_evict_slots`].
fn cache_write_latent_evict(
    cache: &mut KvCache,
    latent: Tensor<4>,
    start_pos: usize,
    sink_tokens: usize,
) {
    let [batch_size, heads, new_seq_len, dim] = latent.dims();
    let slots = alloc_evict_slots(cache, start_pos, new_seq_len, sink_tokens);

    if cache.dtype.is_quantized() {
        let row_len = heads * dim;
        let pack = cache.dtype.pack();
        let block = quant_block(row_len);
        let words = row_len / pack;
        let blocks = row_len / block;
        let rows = latent
            .swap_dims(1, 2)
            .reshape([batch_size * new_seq_len, row_len]);
        let (packed, scales) = quantize_rows(rows, cache.dtype);
        let mut kq = cache.kq.take().expect("mla cache kq buffer");
        let mut ks = cache.ks.take().expect("mla cache ks buffer");
        for (row, &slot) in slots.iter().enumerate() {
            kq = kq.slice_assign(
                [slot..slot + 1, 0..words],
                packed.clone().slice([row..row + 1, 0..words]),
            );
            ks = ks.slice_assign(
                [slot..slot + 1, 0..blocks],
                scales.clone().slice([row..row + 1, 0..blocks]),
            );
        }
        cache.kq = Some(kq);
        cache.ks = Some(ks);
        return;
    }

    let storage = cache_storage_dtype(cache.dtype);
    let latent = if latent.dtype() != storage {
        latent.cast(storage)
    } else {
        latent
    };
    let mut k_buf = cache.k.take().expect("kv cache latent buffer");
    for (row, &slot) in slots.iter().enumerate() {
        let write_slice = [0..batch_size, 0..heads, slot..slot + 1, 0..dim];
        let part = latent
            .clone()
            .slice([0..batch_size, 0..heads, row..row + 1, 0..dim]);
        k_buf = k_buf.slice_assign(write_slice, part);
    }
    cache.k = Some(k_buf);
}

/// Absolute position of every ring slot as an `Int` tensor (`i32::MAX` for
/// empty slots, masked out by the attention).
pub(crate) fn cache_col_positions(cache: &KvCache, device: &Device) -> Tensor<1, Int> {
    let col_pos = cache_positions_i32(cache);
    Tensor::<1, Int>::from_data(TensorData::new(col_pos, [cache.capacity]), device)
}
