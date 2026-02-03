use std::fs;
use std::path::PathBuf;

use anyhow::Result;
use serde::Serialize;

use super::models::{InfraConfig, Provider, ProviderConfig};

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

#[derive(Serialize)]
struct ConfigToSave {
    #[serde(skip_serializing_if = "is_default_provider")]
    default_provider: String,
    #[serde(skip_serializing_if = "ProviderConfigToSave::is_empty")]
    lambda: ProviderConfigToSave,
    #[serde(skip_serializing_if = "ProviderConfigToSave::is_empty")]
    vast: ProviderConfigToSave,
    #[serde(skip_serializing_if = "ProviderConfigToSave::is_empty")]
    prime: ProviderConfigToSave,
}

fn is_default_provider(s: &str) -> bool {
    s == "lambda"
}

#[derive(Serialize, Default)]
struct ProviderConfigToSave {
    #[serde(skip_serializing_if = "Option::is_none")]
    api_key: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    default_region: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    default_instance_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    ssh_key_name: Option<String>,
}

impl ProviderConfigToSave {
    fn is_empty(&self) -> bool {
        self.api_key.is_none()
            && self.default_region.is_none()
            && self.default_instance_type.is_none()
            && self.ssh_key_name.is_none()
    }

    fn from_config(config: &ProviderConfig) -> Self {
        Self {
            api_key: config.api_key.clone(),
            default_region: config.default_region.clone(),
            default_instance_type: config.default_instance_type.clone(),
            ssh_key_name: config.ssh_key_name.clone(),
        }
    }
}

pub fn save_config(config: &InfraConfig) -> Result<()> {
    let path = config_path();

    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }

    let to_save = ConfigToSave {
        default_provider: match config.default_provider {
            Provider::Lambda => "lambda".to_string(),
            Provider::Vast => "vast".to_string(),
            Provider::Prime => "prime".to_string(),
        },
        lambda: ProviderConfigToSave::from_config(&config.lambda_config),
        vast: ProviderConfigToSave::from_config(&config.vast),
        prime: ProviderConfigToSave::from_config(&config.prime),
    };

    let content = toml::to_string_pretty(&to_save)?;
    fs::write(&path, content)?;
    Ok(())
}
