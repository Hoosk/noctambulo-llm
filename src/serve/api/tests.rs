use std::sync::Arc;

use axum::http::{HeaderMap, StatusCode, header};

use crate::serve::state::ServerState;

use super::auth::{check_token, cookie_decode, cookie_encode, cookie_token, query_token};
use super::paths::{is_safe_child_name, is_safe_export_name, is_safe_rel_path, resolve_model_dir};
use super::router;

#[test]
fn test_is_safe_rel_path() {
    assert!(is_safe_rel_path("checkpoints_web"));
    assert!(is_safe_rel_path("checkpoints_web/checkpoint_epoch_1.bin"));
    assert!(is_safe_rel_path("exported/kuro-neko.gguf"));

    assert!(!is_safe_rel_path(""));
    assert!(!is_safe_rel_path(".."));
    assert!(!is_safe_rel_path("../etc/passwd"));
    assert!(!is_safe_rel_path("checkpoints_web/../../etc/passwd"));
    assert!(!is_safe_rel_path("/etc/passwd"));
    assert!(!is_safe_rel_path("."));
}

#[test]
fn test_query_token() {
    assert_eq!(query_token(Some("token=abc")), Some("abc".into()));
    assert_eq!(query_token(Some("a=1&token=xyz&b=2")), Some("xyz".into()));
    assert_eq!(query_token(Some("a=1&b=2")), None);
    assert_eq!(query_token(Some("token=")), Some("".into()));
    assert_eq!(query_token(None), None);
}

#[test]
fn test_is_safe_child_name() {
    assert!(is_safe_child_name("dataset.bin"));
    assert!(is_safe_child_name("TinyStories-train.txt"));
    assert!(is_safe_child_name("model_output_v2_ppl1.83"));

    assert!(!is_safe_child_name(""));
    assert!(!is_safe_child_name("."));
    assert!(!is_safe_child_name(".."));
    assert!(!is_safe_child_name("../evil.txt"));
    assert!(!is_safe_child_name("a/b"));
    assert!(!is_safe_child_name("a\\b"));
    assert!(!is_safe_child_name("/etc/passwd"));
}

#[test]
fn test_is_safe_export_name() {
    assert!(is_safe_export_name("kuro-neko-2026"));
    assert!(is_safe_export_name("model_v3.gguf"));

    assert!(!is_safe_export_name("../evil"));
    assert!(!is_safe_export_name("a/b"));
    assert!(!is_safe_export_name(".hidden"));
    assert!(!is_safe_export_name(""));
    assert!(!is_safe_export_name("model name"));
    assert!(!is_safe_export_name("model$name"));
}

#[test]
fn test_cookie_roundtrip_and_parsing() {
    assert_eq!(cookie_encode("abc-1_2.3~"), "abc-1_2.3~");
    assert_eq!(cookie_encode("a b;c=d"), "a%20b%3Bc%3Dd");
    let encoded = cookie_encode("tok+en/with=weird chars");
    assert_eq!(cookie_decode(&encoded).unwrap(), "tok+en/with=weird chars");
    assert_eq!(cookie_decode("bad%2"), None);
    assert_eq!(cookie_decode("%zz"), None);

    let mut headers = HeaderMap::new();
    headers.insert(header::COOKIE, "a=1; kuro_token=se%20cret; b=2".parse().unwrap());
    assert_eq!(cookie_token(&headers), Some("se cret".into()));

    headers.insert(header::COOKIE, "kuro_token=".parse().unwrap());
    assert_eq!(cookie_token(&headers), Some("".into()));

    headers.insert(header::COOKIE, "other=1".parse().unwrap());
    assert_eq!(cookie_token(&headers), None);
    assert_eq!(cookie_token(&HeaderMap::new()), None);
}

#[test]
fn test_check_token_accepts_header_cookie_and_query() {
    let server = ServerState::new(
        Some("s3cret".into()),
        std::path::PathBuf::from("d"),
        std::path::PathBuf::from("m"),
    );

    assert!(check_token(&server, &HeaderMap::new(), None).is_err());

    let mut headers = HeaderMap::new();
    headers.insert("x-api-token", "s3cret".parse().unwrap());
    assert!(check_token(&server, &headers, None).is_ok());

    let mut headers = HeaderMap::new();
    headers.insert(header::COOKIE, "kuro_token=s3cret".parse().unwrap());
    assert!(check_token(&server, &headers, None).is_ok());

    assert!(check_token(&server, &HeaderMap::new(), Some("token=s3cret")).is_ok());
    assert!(check_token(&server, &HeaderMap::new(), Some("token=nope")).is_err());
}

#[tokio::test]
async fn test_dataset_param_routes_match() {
    use tower::ServiceExt;

    let dir = std::env::temp_dir().join(format!("kuro_routes_test_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("sample.txt"), "hello\nworld\n").unwrap();

    let state = Arc::new(ServerState::new(Some("tok".into()), dir.clone(), dir.clone()));
    let (router, _registry) = router(state);

    let req = axum::http::Request::builder()
        .uri("/api/datasets/info/sample.txt")
        .header("x-api-token", "tok")
        .body(axum::body::Body::empty())
        .unwrap();
    let res = router.clone().oneshot(req).await.unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let body = axum::body::to_bytes(res.into_body(), usize::MAX).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["name"], "sample.txt");
    assert_eq!(json["lines"], 2);

    let req = axum::http::Request::builder()
        .method("DELETE")
        .uri("/api/datasets/delete/sample.txt")
        .header("x-api-token", "tok")
        .body(axum::body::Body::empty())
        .unwrap();
    let res = router.oneshot(req).await.unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    assert!(!dir.join("sample.txt").exists());

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_resolve_model_dir_blocks_traversal() {
    let root = std::env::temp_dir().join(format!("kuro_models_test_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(root.join("model_output")).unwrap();
    std::fs::create_dir_all(root.join("model_output_v2")).unwrap();
    std::fs::create_dir_all(root.parent().unwrap().join("kuro_outside_models")).unwrap();

    assert!(resolve_model_dir(&root, "model_output").is_ok());
    assert!(resolve_model_dir(&root, "").is_ok());
    assert!(resolve_model_dir(&root, "does_not_exist").is_err());
    // Traversal / outside paths must be rejected.
    assert!(resolve_model_dir(&root, "../kuro_outside_models").is_err());
    assert!(resolve_model_dir(&root, "model_output/../../kuro_outside_models").is_err());

    // Absolute path inside the root is accepted (frontend sends full dirs).
    let inside = root.join("model_output_v2");
    assert!(resolve_model_dir(&root, inside.to_str().unwrap()).is_ok());

    let _ = std::fs::remove_dir_all(&root);
    let _ = std::fs::remove_dir_all(root.parent().unwrap().join("kuro_outside_models"));
}
