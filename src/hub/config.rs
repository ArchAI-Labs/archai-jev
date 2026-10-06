//! Explicit configuration of the hub. The core never reads environment variables: the binding
//! reads them once and builds this, so tests cannot touch the real network or the real cache.

use std::path::PathBuf;
use std::time::Duration;

/// Where files are downloaded from unless told otherwise.
pub const DEFAULT_ENDPOINT: &str = "https://huggingface.co";

/// Everything the hub needs to know.
#[derive(Debug, Clone)]
pub struct HubConfig {
    /// Cache root (the layout lives in `<root>/v1`).
    pub cache_root: PathBuf,
    /// Base URL; files are `{endpoint}/{repo}/resolve/{revision}/{path}`.
    pub endpoint: String,
    /// Never open a socket.
    pub offline: bool,
    /// Time allowed to connect.
    pub connect_timeout: Duration,
    /// Time allowed to receive the response headers.
    pub response_timeout: Duration,
    /// Slowest acceptable speed in bytes per second: an attempt gets `remaining / this`
    /// (at least 60 s) to finish. Resuming keeps the progress made.
    pub min_bytes_per_second: u64,
    /// Attempts without progress before giving up.
    pub attempts: usize,
    /// Waits between attempts (the last one repeats).
    pub backoff: Vec<Duration>,
    /// Minimum time between two progress events.
    pub progress_interval: Duration,
    /// Redirects followed per request.
    pub max_redirects: u32,
}

impl HubConfig {
    /// Defaults with the given cache root.
    pub fn new(cache_root: PathBuf) -> Self {
        HubConfig {
            cache_root,
            endpoint: DEFAULT_ENDPOINT.to_string(),
            offline: false,
            connect_timeout: Duration::from_secs(10),
            response_timeout: Duration::from_secs(30),
            min_bytes_per_second: 1 << 20,
            attempts: 3,
            backoff: vec![
                Duration::from_secs(1),
                Duration::from_secs(3),
                Duration::from_secs(9),
            ],
            progress_interval: Duration::from_secs(5),
            max_redirects: 5,
        }
    }

    /// The URL of `path` in `repo` at `revision`.
    pub fn url(&self, repo: &str, revision: &str, path: &str) -> String {
        format!(
            "{}/{repo}/resolve/{revision}/{path}",
            self.endpoint.trim_end_matches('/')
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_endpoint_and_url_scheme() {
        let c = HubConfig::new(PathBuf::from("x"));
        assert_eq!(c.endpoint, "https://huggingface.co");
        assert_eq!(
            c.url("nickprock/archai-jev-qwen-1.5b", "e4f3964a", "model.gguf"),
            "https://huggingface.co/nickprock/archai-jev-qwen-1.5b/resolve/e4f3964a/model.gguf"
        );
        let mut c = c;
        c.endpoint = "http://127.0.0.1:9/".to_string();
        assert_eq!(
            c.url("a/b", "c", "d/e"),
            "http://127.0.0.1:9/a/b/resolve/c/d/e"
        );
    }
}
