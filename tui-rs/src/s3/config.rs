use std::fs;
use std::path::PathBuf;

use anyhow::Result;
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize, Default)]
pub struct S3Config {
    pub bucket: String,
    #[serde(default = "default_prefix")]
    pub prefix: String,
    pub region: Option<String>,
    pub access_key_id: Option<String>,
    pub secret_access_key: Option<String>,
    pub endpoint_url: Option<String>,
}

fn default_prefix() -> String {
    "extty".to_string()
}

pub fn load_config() -> Result<Option<S3Config>> {
    let path = dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".extty")
        .join("s3")
        .join("config.toml");

    if !path.exists() {
        return Ok(None);
    }

    let content = fs::read_to_string(&path)?;
    let config: S3Config = toml::from_str(&content)?;

    if config.bucket.is_empty() {
        return Ok(None);
    }

    Ok(Some(config))
}
