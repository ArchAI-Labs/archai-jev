//! A registry model end to end against the local server: download, verify, cache, offline.

use std::sync::Arc;

use super::Harness;
use crate::error::Error;
use crate::hub::download::UreqTransport;
use crate::models::failures::DownloadFailure;
use crate::models::incompat::Incompat;
use crate::models::load::LoadRequest;
use crate::models::registry::Registry;
use crate::models::resolve::{NameOrPath, Selection};
use crate::models::testing::builder::{Builder, COMMIT};
use crate::models::testing::server::TestServer;

struct Setup {
    h: Harness,
    server: TestServer,
    _fixture: crate::models::testing::Fixture,
}

fn setup() -> Setup {
    let fixture = Builder::tiny().registry().declared(2.0).build();
    let text = std::fs::read_to_string(fixture.manifest_path()).unwrap();
    let index =
        format!(r#"{{"default": "test/tiny@{COMMIT}", "current": {{"test/tiny": "{COMMIT}"}}}}"#);
    let registry = Registry::from_sources(&[text.as_str()], &index).unwrap();
    let server = TestServer::start();
    for file in ["model.gguf", "tokenizer.json"] {
        server.add_file(
            &format!("/test/tiny/resolve/{COMMIT}/{file}"),
            std::fs::read(fixture.path().join(file)).unwrap(),
        );
    }
    let mut h = Harness::with_registry(registry);
    h.endpoint = Some(server.url());
    h.real_transport = Some(Arc::new(UreqTransport::new(&h.hub())));
    Setup {
        h,
        server,
        _fixture: fixture,
    }
}

fn request(name: Option<&str>) -> LoadRequest {
    LoadRequest {
        selection: Selection {
            name_or_path: name.map_or(NameOrPath::Default, |n| NameOrPath::Str(n.to_string())),
            revision: None,
            device: "cpu".to_string(),
            dtype: None,
            manifest: None,
        },
        temperature: None,
        allow_uncalibrated: false,
    }
}

#[test]
fn download_verify_cache_then_no_more_requests() {
    let s = setup();
    // the default model, by name and without arguments
    let model = s.h.load(&request(None)).unwrap();
    assert_eq!(
        (
            model.data.name.as_str(),
            model.data.source,
            model.data.calibrated
        ),
        ("test/tiny", "registry", true)
    );
    assert_eq!(s.server.resolve_requests().len(), 2, "one request per file");
    assert_eq!(s.h.records(), 1);
    let before = s.server.log().len();
    s.h.load(&request(Some("test/tiny"))).unwrap();
    assert_eq!(s.server.log().len(), before, "everything is cached");
    // the blobs are named by their hash
    let blobs: Vec<_> = std::fs::read_dir(s.h.cache.path().join("v1").join("blobs"))
        .unwrap()
        .flatten()
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.len() == 64)
        .collect();
    assert_eq!(blobs.len(), 2);
}

#[test]
fn offline_uses_the_cache_and_never_the_network() {
    let mut s = setup();
    s.h.load(&request(None)).unwrap();
    let before = s.server.log().len();
    s.h.offline = true;
    s.h.load(&request(None)).unwrap();
    assert_eq!(s.server.log().len(), before);
    // an empty cache in offline mode
    let mut empty = setup();
    empty.h.offline = true;
    let err = empty.h.load(&request(None)).err().unwrap();
    assert!(
        matches!(err, Error::ModelDownload(DownloadFailure::Offline { .. })),
        "{err:?}"
    );
    assert_eq!(empty.server.log().len(), 0);
}

#[test]
fn a_tampered_cached_file_is_refused() {
    let s = setup();
    s.h.load(&request(None)).unwrap();
    // flip one byte of a cached blob without changing its size, and forget the record
    let blobs = s.h.cache.path().join("v1").join("blobs");
    let gguf = std::fs::read_dir(&blobs)
        .unwrap()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.file_name().is_some_and(|n| n.len() == 64))
        .max_by_key(|p| std::fs::metadata(p).unwrap().len())
        .unwrap();
    let mut bytes = std::fs::read(&gguf).unwrap();
    let last = bytes.len() - 1;
    bytes[last] ^= 0xFF;
    std::fs::write(&gguf, bytes).unwrap();
    std::fs::remove_dir_all(s.h.cache.path().join("v1").join("verified")).unwrap();
    let err = s.h.load(&request(None)).err().unwrap();
    assert!(
        matches!(err, Error::IncompatibleModel(Incompat::HashMismatch { .. })),
        "{err:?}"
    );
    assert_eq!(s.h.records(), 0);
}

#[test]
fn a_missing_remote_file_is_a_clear_download_error() {
    let s = setup();
    let empty = TestServer::start(); // knows no file at all
    let mut h = Harness::with_registry(s.h.registry.clone());
    h.endpoint = Some(empty.url());
    h.real_transport = Some(Arc::new(UreqTransport::new(&h.hub())));
    let err = h.load(&request(None)).err().unwrap();
    assert!(
        matches!(err, Error::ModelDownload(DownloadFailure::NotFound { .. })),
        "{err:?}"
    );
    assert!(err.to_string().contains("not found"));
    assert_eq!(h.backend.loads(), 0);
}

#[test]
fn revisions_must_be_pinned() {
    let s = setup();
    for bad in ["main", "aaaaaaa", "ffffffff", ""] {
        let mut req = request(Some("test/tiny"));
        req.selection.revision = Some(bad.to_string());
        let err = s.h.load(&req).err().unwrap();
        assert!(
            matches!(
                err,
                Error::IncompatibleModel(Incompat::RevisionNotAccepted { .. })
            ),
            "{bad}: {err:?}"
        );
    }
    assert_eq!(s.server.log().len(), 0);
    // the full hash and an 8-digit prefix are accepted
    for good in [COMMIT, &COMMIT[..8]] {
        let mut req = request(Some("test/tiny"));
        req.selection.revision = Some(good.to_string());
        s.h.load(&req).unwrap();
    }
}
