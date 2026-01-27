use std::fs;
use std::path::PathBuf;

use anyhow::Result;

use super::models::InfraConfig;

fn config_path() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".extty")
        .join("infra")
        .join("config.toml")
}

pub fn load_config() -> Result<InfraConfig> {
    let path = config_path();

    if !path.exists() {
        return Ok(InfraConfig::default());
    }

    let content = fs::read_to_string(&path)?;
    let config: InfraConfig = toml::from_str(&content)?;
    Ok(config)
}
