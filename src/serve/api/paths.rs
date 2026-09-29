use std::path::Path;

use super::error::ApiError;

/// True when `name` is a single, separator-free path component that is safe to
/// join under a root directory (rejects `..`, absolute paths and separators).
pub(super) fn is_safe_child_name(name: &str) -> bool {
    if name.is_empty() || name == "." || name == ".." || name.contains('/') || name.contains('\\') {
        return false;
    }
    Path::new(name)
        .components()
        .all(|c| matches!(c, std::path::Component::Normal(_)))
}

/// Reject path traversal in user-supplied relative paths: no absolute paths,
/// no `..`, no trailing separators. The server resolves them against its own
/// working directory.
pub(super) fn is_safe_rel_path(p: &str) -> bool {
    use std::path::Component;
    !p.is_empty()
        && !Path::new(p).is_absolute()
        && Path::new(p).components().all(|c| matches!(c, Component::Normal(_)))
}

/// Validate the exported GGUF base name so `/api/export` cannot escape
/// `exported/` (e.g. `../foo`) or produce surprising file names.
pub(super) fn is_safe_export_name(name: &str) -> bool {
    is_safe_child_name(name)
        && !name.starts_with('.')
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
}

/// Resolve a user-supplied dataset name to a safe path inside the datasets dir.
/// Only a bare file name (no path separators / traversal) is accepted.
pub(super) fn resolve_dataset(dir: &Path, name: &str) -> Result<std::path::PathBuf, ApiError> {
    if !is_safe_child_name(name) {
        return Err(ApiError::bad_request("invalid dataset path"));
    }

    let cand = dir.join(name);
    if !cand.is_file() {
        return Err(ApiError::not_found(format!("dataset not found: {name}")));
    }
    Ok(cand)
}

/// Validate a new output file name inside the datasets dir (must not exist
/// yet). Rejects traversal so `/api/datasets/combine` cannot write elsewhere.
pub(super) fn resolve_dataset_output(dir: &Path, name: &str) -> Result<std::path::PathBuf, ApiError> {
    if !is_safe_child_name(name) {
        return Err(ApiError::bad_request("invalid output name"));
    }
    Ok(dir.join(name))
}

/// Resolve a model directory request. Bare names are resolved under the models
/// root; explicit paths are accepted only if they canonicalize inside it.
pub(super) fn resolve_model_dir(root: &Path, requested: &str) -> Result<std::path::PathBuf, ApiError> {
    if requested.trim().is_empty() {
        return Ok(root.join("model_output"));
    }

    let candidate = if is_safe_child_name(requested) {
        root.join(requested)
    } else {
        std::path::PathBuf::from(requested)
    };
    let canonical = std::fs::canonicalize(&candidate)
        .map_err(|_| ApiError::not_found(format!("model dir not found: {requested}")))?;
    let canonical_root = std::fs::canonicalize(root).unwrap_or_else(|_| root.to_path_buf());
    if !canonical.starts_with(&canonical_root) {
        return Err(ApiError::forbidden("model dir outside the models directory"));
    }
    Ok(canonical)
}

/// Resolve a model directory for mutating operations (rename/delete/files).
/// Stricter than `resolve_model_dir`: the directory must be an immediate child
/// of the models root, so routes can never touch nested paths (e.g. `src/` when
/// the models dir happens to be the working directory).
pub(super) fn resolve_model_child_dir(root: &Path, requested: &str) -> Result<std::path::PathBuf, ApiError> {
    let candidate = if is_safe_child_name(requested) {
        root.join(requested)
    } else {
        std::path::PathBuf::from(requested)
    };
    let canonical = std::fs::canonicalize(&candidate)
        .map_err(|_| ApiError::not_found(format!("model dir not found: {requested}")))?;
    let canonical_root = std::fs::canonicalize(root).unwrap_or_else(|_| root.to_path_buf());

    if canonical.parent() != Some(canonical_root.as_path()) {
        return Err(ApiError::forbidden("model dir is not a direct child of the models directory"));
    }
    if !canonical.is_dir() {
        return Err(ApiError::bad_request("model dir is not a directory"));
    }
    Ok(canonical)
}
