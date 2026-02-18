use std::collections::HashMap;

use anyhow::{Context, Result, anyhow};
use reqwest::blocking::Client;
use reqwest::header::{AUTHORIZATION, HeaderMap, HeaderValue};
use serde::Deserialize;

use crate::infra::models::{Instance, InstanceStatus, InstanceType, LaunchOptions, Provider};
use crate::infra::providers::CloudProvider;

const BASE_URL: &str = "https://api.primeintellect.ai/api/v1";

/// Parses "root@193.183.22.53 -p 1530" into ("root", Some("193.183.22.53:1530"))
fn parse_ssh_connection(conn: &Option<String>) -> Option<(String, Option<String>)> {
    let conn = conn.as_ref()?;
    let (user_host, port) = if let Some(idx) = conn.find(" -p ") {
        (&conn[..idx], conn[idx + 4..].trim())
    } else {
        (conn.as_str(), "")
    };
    let (user, host) = user_host.split_once('@')?;
    let ip = if port.is_empty() {
        Some(host.to_string())
    } else {
        Some(format!("{}:{}", host, port))
    };
    Some((user.to_string(), ip))
}

fn normalize_status(raw: &str) -> InstanceStatus {
    match raw.to_lowercase().as_str() {
        "pending" | "provisioning" => InstanceStatus::Pending,
        "starting" => InstanceStatus::Booting,
        "running" | "active" => InstanceStatus::Running,
        "stopping" => InstanceStatus::Stopping,
        "stopped" => InstanceStatus::Stopped,
        "terminated" => InstanceStatus::Terminated,
        "failed" | "error" => InstanceStatus::Error,
        _ => InstanceStatus::Error,
    }
}

pub struct PrimeProvider {
    client: Client,
    headers: HeaderMap,
}

impl PrimeProvider {
    pub fn new(api_key: &str) -> Self {
        let mut headers = HeaderMap::new();
        let auth_value = HeaderValue::from_str(&format!("Bearer {}", api_key))
            .expect("Invalid API key for header");
        headers.insert(AUTHORIZATION, auth_value);

        Self {
            client: Client::new(),
            headers,
        }
    }

    fn get<T: for<'de> Deserialize<'de>>(&self, path: &str) -> Result<T> {
        let url = format!("{}{}", BASE_URL, path);
        let response = self
            .client
            .get(&url)
            .headers(self.headers.clone())
            .send()
            .context("Failed to send request")?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().unwrap_or_default();
            return Err(anyhow!("API error {}: {}", status, text));
        }

        let text = response.text().context("Failed to read response body")?;
        serde_json::from_str(&text).with_context(|| format!("Failed to parse response: {}", text))
    }

    fn post<T: for<'de> Deserialize<'de>>(
        &self,
        path: &str,
        body: &serde_json::Value,
    ) -> Result<T> {
        let url = format!("{}{}", BASE_URL, path);
        let response = self
            .client
            .post(&url)
            .headers(self.headers.clone())
            .json(body)
            .send()
            .context("Failed to send request")?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().unwrap_or_default();
            return Err(anyhow!("API error {}: {}", status, text));
        }

        let text = response.text().context("Failed to read response body")?;
        serde_json::from_str(&text).with_context(|| format!("Failed to parse response: {}", text))
    }

    fn delete(&self, path: &str) -> Result<()> {
        let url = format!("{}{}", BASE_URL, path);
        let response = self
            .client
            .delete(&url)
            .headers(self.headers.clone())
            .send()
            .context("Failed to send request")?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().unwrap_or_default();
            return Err(anyhow!("API error {}: {}", status, text));
        }

        Ok(())
    }
}

#[derive(Deserialize)]
struct PodsResponse {
    #[serde(default)]
    data: Vec<PrimePod>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct PrimePod {
    id: serde_json::Value,
    name: Option<String>,
    ip: Option<String>,
    ssh_connection: Option<String>,
    status: Option<String>,
    gpu_name: Option<String>,
    provider_type: Option<String>,
}

impl PrimePod {
    fn id_string(&self) -> String {
        match &self.id {
            serde_json::Value::String(s) => s.clone(),
            serde_json::Value::Number(n) => n.to_string(),
            _ => "unknown".to_string(),
        }
    }
}

#[derive(Deserialize)]
struct GpusResponse {
    #[serde(default)]
    items: Vec<PrimeGpu>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct PrimeGpu {
    gpu_type: Option<String>,
    gpu_count: Option<u32>,
    cloud_id: Option<String>,
    #[serde(default)]
    vcpu: Option<ResourceInfo>,
    #[serde(default)]
    memory: Option<ResourceInfo>,
    #[serde(default)]
    disk: Option<ResourceInfo>,
    gpu_memory: Option<u32>,
    #[serde(default)]
    prices: Option<PriceInfo>,
    region: Option<String>,
    data_center: Option<String>,
    socket: Option<String>,
    provider: Option<String>,
}

#[derive(Deserialize, Default)]
#[serde(rename_all = "camelCase")]
struct ResourceInfo {
    default_count: Option<u32>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct PriceInfo {
    on_demand: Option<f64>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum CreatePodResponse {
    WithId { id: serde_json::Value },
    WithPod { pod: PrimePod },
}

impl CloudProvider for PrimeProvider {
    fn ssh_user(&self) -> &str {
        "root"
    }

    fn list_instances(&self) -> Result<Vec<Instance>> {
        let response: PodsResponse = self.get("/pods/")?;

        Ok(response
            .data
            .into_iter()
            .map(|p| {
                let id = p.id_string();
                let raw_status = p.status.unwrap_or_else(|| "unknown".to_string());

                let (ssh_user, ip) = parse_ssh_connection(&p.ssh_connection)
                    .unwrap_or_else(|| ("root".to_string(), p.ip.clone()));

                Instance {
                    id,
                    name: p.name,
                    ip,
                    status: normalize_status(&raw_status),
                    instance_type: p.gpu_name.unwrap_or_else(|| "unknown".to_string()),
                    region: p.provider_type.unwrap_or_else(|| "unknown".to_string()),
                    provider: Provider::Prime,
                    ssh_user,
                    raw_status,
                    price_cents_per_hour: None,
                }
            })
            .collect())
    }

    fn list_instance_types(&self) -> Result<Vec<InstanceType>> {
        let response: GpusResponse = self.get("/availability/gpus")?;

        Ok(response
            .items
            .into_iter()
            .map(|g| {
                let gpu_type = g.gpu_type.unwrap_or_else(|| "unknown".to_string());
                let gpu_count = g.gpu_count.unwrap_or(1);
                let price_per_hour = g.prices.and_then(|p| p.on_demand).unwrap_or(0.0);
                let region = g.region.unwrap_or_else(|| "unknown".to_string());

                let mut metadata = HashMap::new();
                if let Some(cloud_id) = &g.cloud_id {
                    metadata.insert("cloud_id".to_string(), cloud_id.clone());
                }
                if let Some(socket) = &g.socket {
                    metadata.insert("socket".to_string(), socket.clone());
                }
                if let Some(provider) = &g.provider {
                    metadata.insert("provider".to_string(), provider.clone());
                }
                if let Some(data_center) = &g.data_center {
                    metadata.insert("data_center".to_string(), data_center.clone());
                }
                metadata.insert("gpu_type".to_string(), gpu_type.clone());
                metadata.insert("gpu_count".to_string(), gpu_count.to_string());

                InstanceType {
                    name: format!("{}x{}", gpu_type, gpu_count),
                    description: g.cloud_id,
                    gpu_count,
                    gpu_name: Some(gpu_type),
                    gpu_description: None,
                    gpu_memory_gib: g.gpu_memory.unwrap_or(0),
                    vcpus: g.vcpu.and_then(|r| r.default_count).unwrap_or(0),
                    memory_gib: g.memory.and_then(|r| r.default_count).unwrap_or(0),
                    storage_gib: g.disk.and_then(|r| r.default_count).unwrap_or(0),
                    price_cents_per_hour: (price_per_hour * 100.0) as u32,
                    regions: vec![region],
                    metadata,
                }
            })
            .collect())
    }

    fn launch(&self, opts: &LaunchOptions) -> Result<Vec<String>> {
        let gpu_type = opts
            .metadata
            .get("gpu_type")
            .ok_or_else(|| anyhow!("Missing gpu_type in metadata"))?;
        let gpu_count: u32 = opts
            .metadata
            .get("gpu_count")
            .ok_or_else(|| anyhow!("Missing gpu_count in metadata"))?
            .parse()
            .context("Invalid gpu_count")?;
        let cloud_id = opts
            .metadata
            .get("cloud_id")
            .ok_or_else(|| anyhow!("Missing cloud_id in metadata"))?;
        let socket = opts
            .metadata
            .get("socket")
            .ok_or_else(|| anyhow!("Missing socket in metadata"))?;
        let provider_type = opts
            .metadata
            .get("provider")
            .ok_or_else(|| anyhow!("Missing provider in metadata"))?;

        let data_center_id = opts
            .metadata
            .get("data_center")
            .ok_or_else(|| anyhow!("Missing data_center in metadata"))?;

        let mut pod = serde_json::json!({
            "cloudId": cloud_id,
            "gpuType": gpu_type,
            "socket": socket,
            "gpuCount": gpu_count,
            "dataCenterId": data_center_id,
        });

        if let Some(name) = &opts.name {
            pod["name"] = serde_json::json!(name);
        }
        if let Some(ssh_keys) = &opts.ssh_key_names
            && let Some(key) = ssh_keys.first()
        {
            pod["sshKeyId"] = serde_json::json!(key);
        }

        let body = serde_json::json!({
            "pod": pod,
            "provider": { "type": provider_type },
        });

        let response: CreatePodResponse = self.post("/pods/", &body)?;

        let pod_id = match response {
            CreatePodResponse::WithId { id } => match id {
                serde_json::Value::String(s) => s,
                serde_json::Value::Number(n) => n.to_string(),
                _ => return Err(anyhow!("Invalid pod ID returned")),
            },
            CreatePodResponse::WithPod { pod } => pod.id_string(),
        };

        Ok(vec![pod_id])
    }

    fn terminate(&self, ids: &[String]) -> Result<()> {
        for id in ids {
            self.delete(&format!("/pods/{}", id))?;
        }
        Ok(())
    }
}
