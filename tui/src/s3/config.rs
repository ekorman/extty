use std::fs;
use std::path::PathBuf;

use anyhow::Result;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct S3Config {
    pub bucket: String,
    #[serde(default = "default_prefix")]
    pub prefix: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub region: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub access_key_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub secret_access_key: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub endpoint_url: Option<String>,
}

fn default_prefix() -> String {
    String::new()
}

pub fn config_path() -> PathBuf {
    crate::paths::extty_home().join("s3").join("config.toml")
}

pub fn load_config() -> Result<Option<S3Config>> {
    let path = config_path();

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

pub fn save_config(config: &S3Config) -> Result<()> {
    let path = config_path();

    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }

    let content = toml::to_string_pretty(config)?;
    fs::write(&path, content)?;
    Ok(())
}
