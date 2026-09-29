//! Terminal UI helpers for Kuro-Neko training output.
//!
//! Uses ANSI escape codes and Unicode box-drawing characters
//! for styled terminal output. No external TUI dependencies.

use crate::config::ModelConfig;
use std::io::Write;
use std::path::Path;

/// ANSI color escape codes.
pub mod color {
    pub const RESET: &str = "\x1b[0m";
    pub const BOLD: &str = "\x1b[1m";
    pub const DIM: &str = "\x1b[2m";
    pub const RED: &str = "\x1b[31m";
    pub const CYAN: &str = "\x1b[36m";
    pub const GREEN: &str = "\x1b[32m";
    pub const YELLOW: &str = "\x1b[33m";
    pub const MAGENTA: &str = "\x1b[35m";
    pub const WHITE: &str = "\x1b[97m";
}

/// Width of the header box in terminal columns.
const BOX_W: usize = 77;

// ── Formatting helpers ──────────────────────────────────────────────────────

/// Format a number with comma thousands separators (e.g. `1234567` → `"1,234,567"`).
pub fn format_number(n: usize) -> String {
    let s = n.to_string();
    let mut result = String::new();
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.push(',');
        }
        result.push(c);
    }
    result.chars().rev().collect()
}

/// Render a Unicode progress bar of the given display `width`.
fn progress_bar(pct: f64, width: usize) -> String {
    let clamped = pct.clamp(0.0, 100.0);
    let filled = ((clamped / 100.0) * width as f64) as usize;
    let empty = width.saturating_sub(filled);
    format!(
        "{}{}{}{}{}",
        color::CYAN, "━".repeat(filled),
        color::DIM, "░".repeat(empty),
        color::RESET,
    )
}

// ── Box drawing helpers ─────────────────────────────────────────────────────

fn box_top() {
    println!(
        "{}┌{}┐{}",
        color::CYAN,
        "─".repeat(BOX_W - 2),
        color::RESET,
    );
}

fn box_mid() {
    println!(
        "{}├{}┤{}",
        color::CYAN,
        "─".repeat(BOX_W - 2),
        color::RESET,
    );
}

fn box_bot() {
    println!(
        "{}└{}┘{}",
        color::CYAN,
        "─".repeat(BOX_W - 2),
        color::RESET,
    );
}

/// Print a row inside the box. `visible_len` is the display width of `content`
/// (excluding ANSI escape codes).
fn box_row(content: &str, visible_len: usize) {
    let content_space = BOX_W.saturating_sub(4);
    let right_pad = content_space.saturating_sub(visible_len);
    println!(
        "{}│{}  {}{}{}│{}",
        color::CYAN,
        color::RESET,
        content,
        " ".repeat(right_pad),
        color::CYAN,
        color::RESET,
    );
}

/// Print a labeled row inside the box: `│  Label         value...        │`
fn labeled_row(label: &str, value: &str) {
    let label_w: usize = 16;
    let formatted = format!(
        "{}{}{:<width$}{}{}",
        color::WHITE,
        color::BOLD,
        label,
        color::RESET,
        value,
        width = label_w,
    );
    let vis = label_w + value.chars().count();
    box_row(&formatted, vis);
}

// ── Public display functions ────────────────────────────────────────────────

/// Print a general info message with a cyan arrow prefix.
pub fn print_info(msg: &str) {
    println!(
        "  {}{}▸{} {}",
        color::CYAN,
        color::BOLD,
        color::RESET,
        msg,
    );
}

/// Print a success message with a green checkmark prefix.
pub fn print_success(msg: &str) {
    println!(
        "  {}{}✓{} {}",
        color::GREEN,
        color::BOLD,
        color::RESET,
        msg,
    );
}

/// Print a warning message with a yellow warning sign prefix.
pub fn print_warning(msg: &str) {
    println!(
        "  {}{}⚠{} {}",
        color::YELLOW,
        color::BOLD,
        color::RESET,
        msg,
    );
}

/// Print the training header box with model configuration and training setup.
#[allow(clippy::too_many_arguments)]
pub fn print_training_header(
    config: &ModelConfig,
    num_sequences: usize,
    token_count: usize,
    epochs: usize,
    num_batches: usize,
    batch_size: usize,
    lr: f64,
) {
    // Calculate model parameter count (mirrors the `stats` command formula).
    let n_embd = config.n_embd;
    let n_head = config.n_head;
    let n_kv_head = config.get_n_kv_head();
    let head_dim = n_embd / n_head;
    let kv_dim = n_kv_head * head_dim;
    let n_layer = config.n_layer;
    let vocab_size = config.vocab_size;

    let hidden_dim = ((8 * n_embd) / 3).div_ceil(256) * 256;

    let embed_params = vocab_size * n_embd;
    let q = n_embd * n_embd + n_embd;
    let k = n_embd * kv_dim + kv_dim;
    let v = n_embd * kv_dim + kv_dim;
    let o = n_embd * n_embd;
    let block = n_embd + q + k + v + o + n_embd
        + n_embd * hidden_dim
        + n_embd * hidden_dim
        + hidden_dim * n_embd;
    let total_params = embed_params + n_layer * block + n_embd;
    let fp32_mb = (total_params * 4) as f64 / (1024.0 * 1024.0);

    let params_str = if total_params >= 1_000_000 {
        format!("{:.2}M", total_params as f64 / 1_000_000.0)
    } else if total_params >= 1_000 {
        format!("{:.1}K", total_params as f64 / 1_000.0)
    } else {
        format_number(total_params)
    };

    // Render header box
    println!();
    box_top();

    let title = format!(
        "{}{}KURO-NEKO · Training Engine{}",
        color::BOLD,
        color::CYAN,
        color::RESET,
    );
    box_row(&title, 27);

    box_mid();

    labeled_row("Architecture", "Qwen2.5 Causal Transformer");
    labeled_row(
        "Parameters",
        &format!("{} (FP32: {:.2} MB)", params_str, fp32_mb),
    );
    labeled_row(
        "Context",
        &format!(
            "{} tokens │ Layers: {} │ Heads: {}Q/{}KV",
            config.n_ctx, n_layer, n_head, n_kv_head,
        ),
    );
    labeled_row(
        "Embedding",
        &format!("{} dim │ SwiGLU: {}", n_embd, hidden_dim),
    );
    if config.attn_softcap > 0.0 {
        labeled_row("Softcapping", &format!("{:.1}", config.attn_softcap));
    }
    labeled_row(
        "Dataset",
        &format!(
            "{} sequences │ {} tokens",
            format_number(num_sequences),
            format_number(token_count),
        ),
    );
    labeled_row(
        "Training",
        &format!(
            "{} epochs × {} batches │ Batch: {}",
            epochs, num_batches, batch_size,
        ),
    );
    labeled_row(
        "Optimizer",
        &format!("AdamW │ LR: {:.2e} → cosine │ WD: 0.1", lr),
    );
    labeled_row("Backend", "WGPU (auto)");

    box_bot();
    println!();
}

#[allow(clippy::too_many_arguments)]
pub fn print_training_progress(
    epoch: usize,
    epochs: usize,
    pct: f64,
    batch: usize,
    num_batches: usize,
    batches_per_sec: f64,
    lr: f64,
    loss: f32,
    avg_loss: f32,
    elapsed_min: u64,
    elapsed_s: u64,
    eta_min: u64,
    eta_s: u64,
) {
    let bar = progress_bar(pct, 8);

    // Dynamic color coding for Loss (L)
    let loss_color = if loss >= 7.0 {
        color::RED
    } else if loss >= 4.0 {
        color::YELLOW
    } else {
        color::GREEN
    };

    let avg_loss_color = if avg_loss >= 7.0 {
        color::RED
    } else if avg_loss >= 4.0 {
        color::YELLOW
    } else {
        color::GREEN
    };

    // Dynamic color coding for Learning Rate (LR) - smaller is more converged (optimal)
    let lr_color = if lr >= 2e-4 {
        color::RED
    } else if lr >= 5e-5 {
        color::YELLOW
    } else {
        color::GREEN
    };

    print!(
        "\r  {}Ep {}/{}{}  {} {:5.1}% │ {}/{} │ L:{}{:.2}{}({}{:.2}{}) │ {}{:.1}b/s{} │ LR:{}{:.0e}{} │ {:02}:{:02}<{:02}:{:02}  ",
        color::BOLD, epoch, epochs, color::RESET,
        bar, pct,
        batch, num_batches,
        loss_color, loss, color::RESET,
        avg_loss_color, avg_loss, color::RESET,
        color::CYAN, batches_per_sec, color::RESET,
        lr_color, lr, color::RESET,
        elapsed_min, elapsed_s,
        eta_min, eta_s,
    );
    let _ = std::io::stdout().flush();
}

/// Print a periodic GPU-synced log step with loss information.
#[allow(dead_code, clippy::too_many_arguments)]
pub fn print_log_step(
    epoch: usize,
    epochs: usize,
    batch: usize,
    num_batches: usize,
    progress: f32,
    step_loss: f32,
    avg_loss: f32,
    lr: f64,
) {
    println!(
        "\r  {}Ep {}/{}{} │ Batch {:>4}/{:<4} ({:4.1}%) │ Loss: {}{:.4}{} │ Avg: {}{:.4}{} │ LR {:.0e}      ",
        color::BOLD, epoch, epochs, color::RESET,
        batch, num_batches, progress,
        color::YELLOW, step_loss, color::RESET,
        color::GREEN, avg_loss, color::RESET,
        lr,
    );
}

/// Print epoch completion summary with optional story preview.
pub fn print_epoch_complete(
    epoch: usize,
    epochs: usize,
    avg_loss: f32,
    elapsed: f32,
    preview: Option<&str>,
) {
    let sep = "─".repeat(BOX_W - 2);
    println!(
        "  {}{}{}",
        color::DIM, sep, color::RESET,
    );
    println!(
        "  {}{}✓ Epoch {}/{} Complete{} │ Avg Loss: {}{:.4}{} │ Time: {:.1}s",
        color::GREEN, color::BOLD,
        epoch, epochs,
        color::RESET,
        color::YELLOW, avg_loss, color::RESET,
        elapsed,
    );
    if let Some(text) = preview {
        println!(
            "  {}✨ \"{}\"{}",
            color::MAGENTA,
            text.replace('\n', " "),
            color::RESET,
        );
    }
    println!(
        "  {}{}{}",
        color::DIM, sep, color::RESET,
    );
    println!();
}

/// Print final training completion summary with saved file paths.
pub fn print_training_complete(config_path: &Path, weights_path: &Path) {
    println!(
        "  {}{}✓ Training complete!{}",
        color::GREEN, color::BOLD, color::RESET,
    );
    println!(
        "  {}├─{} Config:  {}",
        color::DIM, color::RESET,
        config_path.display(),
    );
    println!(
        "  {}└─{} Weights: {}",
        color::DIM, color::RESET,
        weights_path.display(),
    );
    println!();
}

/// Print tokenization start message.
pub fn print_tokenization_start(total_bytes: u64) {
    let mb = total_bytes as f64 / 1_048_576.0;
    println!(
        "  {}{}▸{} Tokenizing {}{:.1} MB{} (parallel multi-threaded)...",
        color::CYAN, color::BOLD, color::RESET,
        color::WHITE, mb, color::RESET,
    );
}

/// Print tokenization progress (overwrites current terminal line with `\r`).
#[allow(clippy::too_many_arguments)]
pub fn print_tokenization_progress(
    pct: f64,
    mb_read: f64,
    mb_total: f64,
    mb_per_sec: f64,
    token_count: usize,
    eta_min: u64,
    eta_s: u64,
) {
    let bar = progress_bar(pct, 20);
    print!(
        "\r    {} {:5.1}% │ {:.1}/{:.1} MB │ {}{:.1} MB/s{} │ {} tok │ ETA {:02}:{:02}   ",
        bar, pct,
        mb_read, mb_total,
        color::YELLOW, mb_per_sec, color::RESET,
        format_number(token_count),
        eta_min, eta_s,
    );
    let _ = std::io::stdout().flush();
}

/// Print tokenization completion summary.
pub fn print_tokenization_complete(
    mb_total: f64,
    mb_per_sec: f64,
    token_count: usize,
    elapsed_secs: f64,
) {
    let bar = progress_bar(100.0, 20);
    print!(
        "\r    {} 100.0% │ {:.1} MB │ {:.1} MB/s │ {} tokens                                  \n",
        bar, mb_total, mb_per_sec, format_number(token_count),
    );
    let _ = std::io::stdout().flush();
    println!(
        "  {}{}✓{} Tokenization complete in {:.2}s — {} tokens loaded.",
        color::GREEN, color::BOLD, color::RESET,
        elapsed_secs, format_number(token_count),
    );
}
