use std::fs;
use std::path::PathBuf;

use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use super::models::{InstanceStatus, Provider};

fn state_path() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".extty")
        .join("infra")
        .join("instances.json")
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstanceState {
    pub instance_id: String,
    pub provider: Provider,
    pub name: Option<String>,
    pub created_at: DateTime<Utc>,
    pub instance_type: String,
    pub region: String,
    pub ip: Option<String>,
    pub status: InstanceStatus,
    pub ssh_user: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct InstanceStateFile {
    pub instances: Vec<InstanceState>,
}

pub fn load_state() -> Result<InstanceStateFile> {
    let path = state_path();

    if !path.exists() {
        return Ok(InstanceStateFile::default());
    }

    let content = fs::read_to_string(&path)?;
    let state: InstanceStateFile = serde_json::from_str(&content)?;
    Ok(state)
}

pub fn save_state(state: &InstanceStateFile) -> Result<()> {
    let path = state_path();

    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }

    let content = serde_json::to_string_pretty(state)?;
    fs::write(&path, content)?;
    Ok(())
}
