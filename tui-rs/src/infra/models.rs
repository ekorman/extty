use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Provider {
    #[default]
    Lambda,
    Vast,
    Prime,
}

impl Provider {
    pub fn display_name(&self) -> &'static str {
        match self {
            Provider::Lambda => "Lambda",
            Provider::Vast => "Vast.ai",
            Provider::Prime => "Prime",
        }
    }

    pub fn all() -> &'static [Provider] {
        &[Provider::Lambda, Provider::Vast, Provider::Prime]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum InstanceStatus {
    Pending,
    Booting,
    Running,
    Stopping,
    Stopped,
    Terminated,
    Error,
}

impl InstanceStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            InstanceStatus::Pending => "pending",
            InstanceStatus::Booting => "booting",
            InstanceStatus::Running => "running",
            InstanceStatus::Stopping => "stopping",
            InstanceStatus::Stopped => "stopped",
            InstanceStatus::Terminated => "terminated",
            InstanceStatus::Error => "error",
        }
    }

    pub fn is_active(&self) -> bool {
        matches!(
            self,
            InstanceStatus::Pending | InstanceStatus::Booting | InstanceStatus::Running
        )
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Instance {
    pub id: String,
    pub name: Option<String>,
    pub ip: Option<String>,
    pub status: InstanceStatus,
    pub instance_type: String,
    pub region: String,
    pub provider: Provider,
    pub ssh_user: String,
    pub raw_status: String,
}

impl Instance {
    pub fn display_name(&self) -> &str {
        self.name.as_deref().unwrap_or(&self.id)
    }

    #[allow(dead_code)]
    pub fn ssh_command(&self) -> Option<String> {
        self.ip.as_ref().map(|ip| {
            if ip.contains(':') {
                let parts: Vec<&str> = ip.split(':').collect();
                format!("ssh -p {} {}@{}", parts[1], self.ssh_user, parts[0])
            } else {
                format!("ssh {}@{}", self.ssh_user, ip)
            }
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstanceType {
    pub name: String,
    pub description: Option<String>,
    pub gpu_count: u32,
    pub gpu_name: Option<String>,
    pub gpu_description: Option<String>,
    pub vcpus: u32,
    pub memory_gib: u32,
    pub storage_gib: u32,
    pub price_cents_per_hour: u32,
    pub regions: Vec<String>,
}

impl InstanceType {
    pub fn price_display(&self) -> String {
        let dollars = self.price_cents_per_hour as f64 / 100.0;
        format!("${:.2}/hr", dollars)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ProviderConfig {
    pub api_key: Option<String>,
    pub default_region: Option<String>,
    pub default_instance_type: Option<String>,
    pub ssh_key_name: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InfraConfig {
    #[serde(default)]
    pub default_provider: Provider,
    #[serde(default, alias = "lambda")]
    pub lambda_config: ProviderConfig,
    #[serde(default)]
    pub vast: ProviderConfig,
    #[serde(default)]
    pub prime: ProviderConfig,
    #[serde(default)]
    pub api_key: Option<String>,
    #[serde(default)]
    pub default_region: Option<String>,
    #[serde(default)]
    pub default_instance_type: Option<String>,
    #[serde(default)]
    pub ssh_key_name: Option<String>,
}

impl Default for InfraConfig {
    fn default() -> Self {
        Self {
            default_provider: Provider::Lambda,
            lambda_config: ProviderConfig::default(),
            vast: ProviderConfig::default(),
            prime: ProviderConfig::default(),
            api_key: None,
            default_region: None,
            default_instance_type: None,
            ssh_key_name: None,
        }
    }
}

impl InfraConfig {
    pub fn get_provider_config(&self, provider: Provider) -> ProviderConfig {
        match provider {
            Provider::Lambda => {
                let config = &self.lambda_config;
                ProviderConfig {
                    api_key: config.api_key.clone().or_else(|| self.api_key.clone()),
                    default_region: config
                        .default_region
                        .clone()
                        .or_else(|| self.default_region.clone()),
                    default_instance_type: config
                        .default_instance_type
                        .clone()
                        .or_else(|| self.default_instance_type.clone()),
                    ssh_key_name: config
                        .ssh_key_name
                        .clone()
                        .or_else(|| self.ssh_key_name.clone()),
                }
            }
            Provider::Vast => self.vast.clone(),
            Provider::Prime => self.prime.clone(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct LaunchOptions {
    pub instance_type: String,
    pub region: Option<String>,
    pub ssh_key_names: Option<Vec<String>>,
    pub name: Option<String>,
}
