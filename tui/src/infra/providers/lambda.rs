use std::collections::HashMap;

use anyhow::{Context, Result, anyhow};
use reqwest::blocking::Client;
use reqwest::header::{AUTHORIZATION, HeaderMap, HeaderValue};
use serde::Deserialize;

use crate::infra::models::{Instance, InstanceStatus, InstanceType, LaunchOptions, Provider};
use crate::infra::providers::CloudProvider;

const BASE_URL: &str = "https://cloud.lambdalabs.com/api/v1";

fn normalize_status(raw: &str) -> InstanceStatus {
    match raw {
        "booting" => InstanceStatus::Booting,
        "active" => InstanceStatus::Running,
        "unhealthy" => InstanceStatus::Error,
        "terminated" => InstanceStatus::Terminated,
        "terminating" => InstanceStatus::Stopping,
        "preempted" => InstanceStatus::Terminated,
        _ => InstanceStatus::Error,
    }
}

fn parse_gpu_memory(desc: &Option<String>) -> u32 {
    let Some(desc) = desc else { return 0 };
    for part in desc.split_whitespace() {
        if let Some(num_str) = part.strip_suffix("GB")
            && let Ok(num) = num_str.parse::<u32>()
        {
            return num;
        }
    }
    0
}

pub struct LambdaProvider {
    client: Client,
    headers: HeaderMap,
}

impl LambdaProvider {
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

        response.json().context("Failed to parse response")
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

        response.json().context("Failed to parse response")
    }
}

#[derive(Deserialize)]
struct InstancesResponse {
    data: Vec<LambdaInstance>,
}

#[derive(Deserialize)]
struct LambdaInstance {
    id: String,
    name: Option<String>,
    ip: Option<String>,
    status: String,
    instance_type: LambdaInstanceType,
    region: LambdaRegion,
}

#[derive(Deserialize)]
struct LambdaInstanceType {
    name: String,
    description: Option<String>,
    gpu_description: Option<String>,
    #[serde(default)]
    specs: LambdaSpecs,
    #[serde(default)]
    price_cents_per_hour: u32,
}

#[derive(Deserialize, Default)]
struct LambdaSpecs {
    #[serde(default)]
    gpus: u32,
    #[serde(default)]
    vcpus: u32,
    #[serde(default)]
    memory_gib: u32,
    #[serde(default)]
    storage_gib: u32,
}

#[derive(Deserialize, Clone)]
struct LambdaRegion {
    name: String,
}

#[derive(Deserialize)]
struct InstanceTypesResponse {
    data: HashMap<String, InstanceTypeWrapper>,
}

#[derive(Deserialize)]
struct InstanceTypeWrapper {
    instance_type: LambdaInstanceType,
    #[serde(default)]
    regions_with_capacity_available: Vec<LambdaRegion>,
}

#[derive(Deserialize)]
struct LaunchResponse {
    data: LaunchData,
}

#[derive(Deserialize)]
struct LaunchData {
    instance_ids: Vec<String>,
}

impl CloudProvider for LambdaProvider {
    fn ssh_user(&self) -> &str {
        "ubuntu"
    }

    fn list_instances(&self) -> Result<Vec<Instance>> {
        let response: InstancesResponse = self.get("/instances")?;

        Ok(response
            .data
            .into_iter()
            .map(|i| Instance {
                id: i.id,
                name: i.name,
                ip: i.ip,
                status: normalize_status(&i.status),
                instance_type: i.instance_type.name,
                region: i.region.name,
                provider: Provider::Lambda,
                ssh_user: "ubuntu".to_string(),
                raw_status: i.status,
            })
            .collect())
    }

    fn list_instance_types(&self) -> Result<Vec<InstanceType>> {
        let response: InstanceTypesResponse = self.get("/instance-types")?;

        Ok(response
            .data
            .into_iter()
            .map(|(name, wrapper)| {
                let it = wrapper.instance_type;
                let gpu_memory_gib = parse_gpu_memory(&it.gpu_description);
                InstanceType {
                    name,
                    description: it.description,
                    gpu_count: it.specs.gpus,
                    gpu_name: None,
                    gpu_description: it.gpu_description,
                    gpu_memory_gib,
                    vcpus: it.specs.vcpus,
                    memory_gib: it.specs.memory_gib,
                    storage_gib: it.specs.storage_gib,
                    price_cents_per_hour: it.price_cents_per_hour,
                    regions: wrapper
                        .regions_with_capacity_available
                        .into_iter()
                        .map(|r| r.name)
                        .collect(),
                    metadata: Default::default(),
                }
            })
            .collect())
    }

    fn launch(&self, opts: &LaunchOptions) -> Result<Vec<String>> {
        let region = opts
            .region
            .as_ref()
            .ok_or_else(|| anyhow!("Region is required for Lambda Cloud"))?;
        let ssh_keys = opts
            .ssh_key_names
            .as_ref()
            .ok_or_else(|| anyhow!("SSH key is required for Lambda Cloud"))?;

        let mut body = serde_json::json!({
            "region_name": region,
            "instance_type_name": opts.instance_type,
            "ssh_key_names": ssh_keys,
        });

        if let Some(name) = &opts.name {
            body["name"] = serde_json::json!(name);
        }

        let response: LaunchResponse = self.post("/instance-operations/launch", &body)?;
        Ok(response.data.instance_ids)
    }

    fn terminate(&self, ids: &[String]) -> Result<()> {
        let body = serde_json::json!({ "instance_ids": ids });
        let _: serde_json::Value = self.post("/instance-operations/terminate", &body)?;
        Ok(())
    }
}
