//! The decoder-only model and its configuration-driven initialization.

use burn::module::{Module, ModuleMapper, Param, ParamId};
use burn::nn::Embedding;
use burn::tensor::{DType, Device, FloatDType, Int, Tensor};

use crate::config::ModelConfig;

use super::cache::{
    KvCache, cache_col_positions, cache_positions_i32, cache_write, cache_write_latent,
    sample_keep_mask,
};
use super::layers::{Block, BlockConfig, KvProj, KvProjConfig, MlaProj, MlaProjConfig, RmsNorm};
use super::quant::{KvCacheDtype, dequantize_rows};
use super::rope::extended_ctx_len;

/// Module mapper that casts every float parameter to the requested dtype.
struct CastFloatDtype(FloatDType);

impl ModuleMapper for CastFloatDtype {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<D>>) -> Param<Tensor<D>> {
        param.map(|tensor| tensor.cast(self.0))
    }
}

#[derive(Module, Debug)]
pub struct Llm {
    pub embed_tokens: Embedding,
    /// One K/V projection per sharing group (`ceil(n_layer / kv_share_group)`).
    /// Empty when MLA is enabled.
    pub kv: Vec<KvProj>,
    /// One MLA projection per sharing group; empty unless `mla_latent_dim` is
    /// set. Only the compressed latent is cached.
    pub mla: Vec<MlaProj>,
    pub blocks: Vec<Block>,
    pub norm: RmsNorm,
    pub n_ctx: usize,
    /// Position limit: `n_ctx` extended by the active RoPE scaling factor.
    pub max_seq: usize,
    /// Layers per K/V group (`1` = no cross-layer reuse).
    pub kv_share_group: usize,
    /// Heavy-hitter eviction budget in rows (`None` = dense/ring cache). The
    /// training forward uses it to build the eviction keep-mask.
    pub evict_budget: Option<usize>,
    /// Recent tokens protected from eviction (training mask).
    pub evict_recent: usize,
}

impl Llm {
    /// Total number of positions the model can attend to (the config `n_ctx`
    /// extended by the active RoPE scaling factor).
    pub fn max_seq_len(&self) -> usize {
        self.max_seq
    }

    /// Whether the model compresses K/V into an MLA latent.
    pub fn use_mla(&self) -> bool {
        !self.mla.is_empty()
    }

    /// Number of attention sinks configured on the blocks.
    pub fn sink_tokens(&self) -> usize {
        self.blocks.first().map(|b| b.attn.sink_tokens).unwrap_or(0)
    }

    /// Number of rows each layer cache needs. With a sliding window the cache
    /// is bounded to `window + sinks`; otherwise it spans the whole context.
    pub fn kv_capacity(&self) -> usize {
        match self.blocks.first() {
            Some(b) => match b.attn.sliding_window {
                Some(window) => window + b.attn.sink_tokens,
                None => self.max_seq,
            },
            None => self.max_seq,
        }
    }

    /// Index of the K/V group that serves `layer`.
    fn group_of(&self, layer: usize) -> usize {
        layer / self.kv_share_group.max(1)
    }

    /// Casts every float parameter to `dtype` for f16 inference. Records are
    /// stored in f32, so this only happens in memory after loading.
    pub fn cast_compute_dtype(self, dtype: FloatDType) -> Self {
        self.map(&mut CastFloatDtype(dtype))
    }

    /// Preallocated caches (one per K/V group) sized to `capacity` positions.
    /// MLA groups cache only the compressed latent. Quantized dtypes store
    /// packed integer words plus per-block scales and dequantize on read.
    pub fn new_kv_caches(
        &self,
        capacity: usize,
        dtype: KvCacheDtype,
        device: &Device,
    ) -> Vec<KvCache> {
        if self.use_mla() {
            self.mla
                .iter()
                .map(|m| {
                    KvCache::preallocated_single(1, 1, m.d_latent, capacity, dtype, device)
                })
                .collect()
        } else {
            self.kv
                .iter()
                .map(|kv| {
                    let head_dim = kv.n_embd / kv.n_head;
                    KvCache::preallocated(1, kv.n_kv_head, head_dim, capacity, dtype, device)
                })
                .collect()
        }
    }

    /// Forward pass over `input`. A single-token input is padded to two rows
    /// before the matmuls because the autotuned M=1 GEMV kernel returns wrong
    /// results on WGPU for output widths >= 32; only the first output row is
    /// returned.
    pub fn forward(&self, input: Tensor<2, Int>) -> Tensor<3> {
        let [_, seq_len] = input.dims();
        if seq_len == 1 {
            let input2 = Tensor::cat(vec![input.clone(), input], 1);
            let out = self.forward_impl(input2);
            let [_, _, vocab] = out.dims();
            return out.slice([0..1, 0..1, 0..vocab]);
        }
        self.forward_impl(input)
    }

    fn forward_impl(&self, input: Tensor<2, Int>) -> Tensor<3> {
        let tok_emb = self.embed_tokens.forward(input);
        let [_, seq_len, _] = tok_emb.dims();
        let device = tok_emb.device();
        let row_pos = Tensor::<1, Int>::arange(0..seq_len as i64, &device);
        let col_pos = Tensor::<1, Int>::arange(0..seq_len as i64, &device);
        let group = self.kv_share_group.max(1);
        let positions: Vec<i32> = (0..seq_len as i32).collect();
        // Eviction-aligned training: sample this batch's heavy-hitter columns.
        let keep_old = self.evict_budget.map(|budget| {
            sample_keep_mask(
                seq_len,
                self.n_ctx,
                budget,
                self.evict_recent,
                self.sink_tokens(),
                &device,
            )
        });

        let mut x = tok_emb;
        let mut group_kv: Option<(Tensor<4>, Tensor<4>)> = None;
        for (i, block) in self.blocks.iter().enumerate() {
            if i % group == 0 {
                let ln = block.input_layernorm.forward(x.clone());
                group_kv = Some(if self.use_mla() {
                    let latent = self.mla[self.group_of(i)].compress(ln);
                    self.mla[self.group_of(i)].expand(latent, &positions)
                } else {
                    self.kv[self.group_of(i)].forward(ln, 0)
                });
            }
            let (k, v) = group_kv.as_ref().expect("group leader computed K/V");
            let (x_new, _) = block.forward(
                x,
                k.clone(),
                v.clone(),
                0,
                row_pos.clone(),
                col_pos.clone(),
                keep_old.clone(),
                false,
            );
            x = x_new;
        }

        let x = self.norm.forward(x);
        // Qwen Weight Tying: tie_word_embeddings = true
        let weight = self.embed_tokens.weight.val().transpose().unsqueeze_dim::<3>(0);
        x.matmul(weight)
    }

    /// Forward pass with a KV cache. A single-token input is padded to two
    /// rows before the matmuls (see [`Llm::forward`]); only the first row is
    /// written to the cache and returned.
    pub fn forward_with_cache(
        &self,
        input: Tensor<2, Int>,
        start_pos: usize,
        caches: &mut [KvCache],
    ) -> Tensor<3> {
        let [_, seq_len] = input.dims();
        if seq_len == 1 {
            let input2 = Tensor::cat(vec![input.clone(), input], 1);
            let out = self.forward_with_cache_impl(input2, start_pos, caches, 1);
            let [_, _, vocab] = out.dims();
            return out.slice([0..1, 0..1, 0..vocab]);
        }
        self.forward_with_cache_impl(input, start_pos, caches, seq_len)
    }

    fn forward_with_cache_impl(
        &self,
        input: Tensor<2, Int>,
        start_pos: usize,
        caches: &mut [KvCache],
        cache_rows: usize,
    ) -> Tensor<3> {
        let tok_emb = self.embed_tokens.forward(input);
        let [_, seq_len, _] = tok_emb.dims();
        let device = tok_emb.device();
        let row_pos = Tensor::<1, Int>::arange(
            start_pos as i64..(start_pos + seq_len) as i64,
            &device,
        );
        let group = self.kv_share_group.max(1);
        // Cache rows may be stored in f32/f16 or dequantized to f32; attention
        // always runs in the compute dtype (the embedding output dtype).
        let compute_dtype = tok_emb.dtype();

        let mut x = tok_emb;
        for (i, block) in self.blocks.iter().enumerate() {
            let g = self.group_of(i);
            if i % group == 0 {
                let ln = block.input_layernorm.forward(x.clone());
                if self.use_mla() {
                    let latent = self.mla[g].compress(ln);
                    let latent = latent.slice([
                        0..1,
                        0..1,
                        0..cache_rows,
                        0..self.mla[g].d_latent,
                    ]);
                    cache_write_latent(&mut caches[g], latent, start_pos, block.attn.sink_tokens);
                } else {
                    let kv = &self.kv[g];
                    let head_dim = kv.n_embd / kv.n_head;
                    let (k, v) = kv.forward(ln, start_pos);
                    let k = k.slice([0..1, 0..kv.n_kv_head, 0..cache_rows, 0..head_dim]);
                    let v = v.slice([0..1, 0..kv.n_kv_head, 0..cache_rows, 0..head_dim]);
                    cache_write(&mut caches[g], k, v, start_pos, block.attn.sink_tokens);
                }
            }
            let cache = &caches[g];
            let (k_full, v_full) = if self.use_mla() {
                let latent = if cache.dtype.is_quantized() {
                    dequantize_rows(
                        cache.kq.clone().expect("mla cache kq buffer"),
                        cache.ks.clone().expect("mla cache ks buffer"),
                        cache.dtype,
                    )
                    .reshape([1, 1, cache.capacity, self.mla[g].d_latent])
                } else {
                    cache.k.clone().expect("mla cache latent buffer")
                };
                let latent = if latent.dtype() != compute_dtype {
                    latent.cast(compute_dtype)
                } else {
                    latent
                };
                let positions = cache_positions_i32(cache);
                self.mla[g].expand(latent, &positions)
            } else if cache.dtype.is_quantized() {
                let kv = &self.kv[g];
                let head_dim = kv.n_embd / kv.n_head;
                // Packed rows are slot-major with (head, dim) flattened, so
                // reshape through [capacity, head, dim] before moving the head
                // axis in front of the sequence axis.
                let k = dequantize_rows(
                    cache.kq.clone().expect("kv cache kq buffer"),
                    cache.ks.clone().expect("kv cache ks buffer"),
                    cache.dtype,
                )
                .reshape([cache.capacity, kv.n_kv_head, head_dim])
                .swap_dims(0, 1)
                .unsqueeze_dim::<4>(0);
                let v = dequantize_rows(
                    cache.vq.clone().expect("kv cache vq buffer"),
                    cache.vs.clone().expect("kv cache vs buffer"),
                    cache.dtype,
                )
                .reshape([cache.capacity, kv.n_kv_head, head_dim])
                .swap_dims(0, 1)
                .unsqueeze_dim::<4>(0);
                (k, v)
            } else {
                let k = cache.k.clone().expect("kv cache k buffer");
                let v = cache.v.clone().expect("kv cache v buffer");
                (k, v)
            };
            let to_compute = |t: Tensor<4>| {
                if t.dtype() != compute_dtype {
                    t.cast(compute_dtype)
                } else {
                    t
                }
            };
            let (k_full, v_full) = (to_compute(k_full), to_compute(v_full));
            let col_pos = cache_col_positions(cache, &device);
            let want_mass = cache.eviction_enabled();
            let (x_new, mass) = block.forward(
                x,
                k_full,
                v_full,
                start_pos,
                row_pos.clone(),
                col_pos,
                None,
                want_mass,
            );
            x = x_new;
            if let Some(mass) = mass {
                let scores = caches[g].scores.take().expect("eviction scores enabled");
                caches[g].scores = Some(scores + mass.cast(FloatDType::F32));
            }
        }

        let x = self.norm.forward(x);
        // Qwen Weight Tying: tie_word_embeddings = true
        let weight = self.embed_tokens.weight.val().transpose().unsqueeze_dim::<3>(0);
        x.matmul(weight)
    }
}

impl ModelConfig {
    pub fn init_model(&self, device: &Device) -> Llm {
        let kv_head = self.get_n_kv_head();
        let rope_kind = self.rope_kind();
        let rope_factor = self.rope_factor;
        let rope_base = self.rope_base;
        let hidden = self.ffn_hidden();
        let max_seq = extended_ctx_len(self.n_ctx, rope_kind, rope_factor);
        // Invalid eviction configs are rejected by the CLI/API entry points;
        // here they only downgrade to the dense mask.
        let eviction = self.eviction_geometry().ok().flatten();
        let blocks = (0..self.n_layer)
            .map(|_| {
                BlockConfig {
                    n_embd: self.n_embd,
                    n_head: self.n_head,
                    n_kv_head: kv_head,
                    n_ctx: self.n_ctx,
                    dropout: self.dropout,
                    attn_softcap: self.attn_softcap,
                    rope_kind,
                    rope_factor,
                    rope_base,
                    sliding_window: self.sliding_window,
                    sink_tokens: self.sink_tokens,
                    evict_budget: eviction.map(|e| e.budget),
                    evict_recent: eviction.map(|e| e.recent).unwrap_or(0),
                    hidden,
                    rms_norm_eps: self.rms_norm_eps,
                }
                .init(device)
            })
            .collect();
        let kv: Vec<KvProj> = if self.use_mla() {
            Vec::new()
        } else {
            (0..self.n_kv_groups())
                .map(|_| {
                    KvProjConfig {
                        n_embd: self.n_embd,
                        n_head: self.n_head,
                        n_kv_head: kv_head,
                        n_ctx: self.n_ctx,
                        rope_kind,
                        rope_factor,
                        rope_base,
                    }
                    .init(device)
                })
                .collect()
        };
        let mla: Vec<MlaProj> = if self.use_mla() {
            (0..self.n_kv_groups())
                .map(|_| {
                    MlaProjConfig {
                        n_embd: self.n_embd,
                        n_head: self.n_head,
                        n_kv_head: kv_head,
                        n_ctx: self.n_ctx,
                        rope_kind,
                        rope_factor,
                        rope_base,
                        d_latent: self.mla_dim(),
                    }
                    .init(device)
                })
                .collect()
        } else {
            Vec::new()
        };

        // Qwen Normal(0, 0.02) weight initialization for tied word embeddings
        let std = 0.02f64;
        let mut embed_weight = Tensor::<2>::random(
            [self.vocab_size, self.n_embd],
            burn::tensor::Distribution::Normal(0.0, std),
            (device, DType::F32),
        );
        if device.is_autodiff() {
            embed_weight = embed_weight.require_grad();
        }
        let embed_tokens = Embedding {
            weight: Param::initialized(ParamId::new(), embed_weight),
        };

        Llm {
            embed_tokens,
            kv,
            mla,
            blocks,
            norm: RmsNorm::new(self.n_embd, self.rms_norm_eps, device),
            n_ctx: self.n_ctx,
            max_seq,
            kv_share_group: self.kv_share_group.max(1),
            evict_budget: eviction.map(|e| e.budget),
            evict_recent: eviction.map(|e| e.recent).unwrap_or(0),
        }
    }
}
