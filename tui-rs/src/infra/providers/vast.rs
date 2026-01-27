use std::collections::HashSet;

use anyhow::{Context, Result, anyhow};
use reqwest::blocking::Client;
use reqwest::header::{AUTHORIZATION, HeaderMap, HeaderValue};
use serde::Deserialize;

use crate::infra::models::{Instance, InstanceStatus, InstanceType, LaunchOptions, Provider};
use crate::infra::providers::CloudProvider;

const BASE_URL: &str = "https://console.vast.ai/api/v0";

fn normalize_status(raw: &str) -> InstanceStatus {
    match raw.to_lowercase().as_str() {
        "running" => InstanceStatus::Running,
        "loading" => InstanceStatus::Booting,
        "created" => InstanceStatus::Pending,
        "exited" => InstanceStatus::Stopped,
        "destroying" => InstanceStatus::Stopping,
        "destroyed" => InstanceStatus::Terminated,
        "offline" => InstanceStatus::Error,
        _ => InstanceStatus::Error,
    }
}

pub struct VastProvider {
    client: Client,
    headers: HeaderMap,
}

impl VastProvider {
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

    fn put<T: for<'de> Deserialize<'de>>(&self, path: &str, body: &serde_json::Value) -> Result<T> {
        let url = format!("{}{}", BASE_URL, path);
        let response = self
            .client
            .put(&url)
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
struct InstancesResponse {
    #[serde(default)]
    instances: Vec<VastInstance>,
}

#[derive(Deserialize)]
struct VastInstance {
    id: i64,
    label: Option<String>,
    ssh_host: Option<String>,
    ssh_port: Option<u16>,
    actual_status: Option<String>,
    status_msg: Option<String>,
    gpu_name: Option<String>,
    geolocation: Option<String>,
}

#[derive(Deserialize)]
struct BundlesResponse {
    #[serde(default)]
    offers: Vec<VastOffer>,
}

#[derive(Deserialize)]
struct VastOffer {
    id: i64,
    gpu_name: Option<String>,
    num_gpus: Option<u32>,
    cpu_cores_effective: Option<f64>,
    cpu_ram: Option<f64>,
    disk_space: Option<f64>,
    dph_base: Option<f64>,
    geolocation: Option<String>,
}

#[derive(Deserialize)]
struct CreateResponse {
    new_contract: Option<i64>,
}

impl CloudProvider for VastProvider {
    fn name(&self) -> Provider {
        Provider::Vast
    }

    fn ssh_user(&self) -> &str {
        "root"
    }

    fn list_instances(&self) -> Result<Vec<Instance>> {
        let response: InstancesResponse = self.get("/instances/")?;

        Ok(response
            .instances
            .into_iter()
            .map(|i| {
                let raw_status = i
                    .actual_status
                    .or(i.status_msg)
                    .unwrap_or_else(|| "unknown".to_string());
                let ip = i.ssh_host.map(|host| {
                    let port = i.ssh_port.unwrap_or(22);
                    format!("{}:{}", host, port)
                });

                Instance {
                    id: i.id.to_string(),
                    name: i.label,
                    ip,
                    status: normalize_status(&raw_status),
                    instance_type: i.gpu_name.unwrap_or_else(|| "unknown".to_string()),
                    region: i.geolocation.unwrap_or_else(|| "unknown".to_string()),
                    provider: Provider::Vast,
                    ssh_user: "root".to_string(),
                    raw_status,
                }
            })
            .collect())
    }

    fn get_instance(&self, id: &str) -> Result<Instance> {
        let response: InstancesResponse = self.get(&format!("/instances/?id={}", id))?;

        let instance = response
            .instances
            .into_iter()
            .find(|i| i.id.to_string() == id)
            .ok_or_else(|| anyhow!("Instance {} not found", id))?;

        let raw_status = instance
            .actual_status
            .or(instance.status_msg)
            .unwrap_or_else(|| "unknown".to_string());
        let ip = instance.ssh_host.map(|host| {
            let port = instance.ssh_port.unwrap_or(22);
            format!("{}:{}", host, port)
        });

        Ok(Instance {
            id: instance.id.to_string(),
            name: instance.label,
            ip,
            status: normalize_status(&raw_status),
            instance_type: instance.gpu_name.unwrap_or_else(|| "unknown".to_string()),
            region: instance
                .geolocation
                .unwrap_or_else(|| "unknown".to_string()),
            provider: Provider::Vast,
            ssh_user: "root".to_string(),
            raw_status,
        })
    }

    fn list_instance_types(&self) -> Result<Vec<InstanceType>> {
        let body = serde_json::json!({
            "verified": {"eq": true},
            "external": {"eq": false},
            "rentable": {"eq": true},
            "num_gpus": {"gte": 1},
            "type": "on-demand",
            "order": [["dph_total", "asc"]],
            "limit": 100,
        });

        let response: BundlesResponse = self.post("/bundles/", &body)?;

        let mut seen: HashSet<String> = HashSet::new();
        let mut types = Vec::new();

        for offer in response.offers {
            let gpu_name = offer.gpu_name.unwrap_or_else(|| "unknown".to_string());
            let num_gpus = offer.num_gpus.unwrap_or(1);
            let type_key = format!("{}x{}", gpu_name, num_gpus);

            if seen.contains(&type_key) {
                continue;
            }
            seen.insert(type_key.clone());

            let price_per_gpu = offer.dph_base.unwrap_or(0.0);
            let total_price = price_per_gpu * num_gpus as f64;

            types.push(InstanceType {
                name: type_key,
                description: Some(format!("{}x {}", num_gpus, gpu_name)),
                gpu_count: num_gpus,
                gpu_name: Some(gpu_name),
                vcpus: offer.cpu_cores_effective.unwrap_or(0.0) as u32,
                memory_gib: (offer.cpu_ram.unwrap_or(0.0) / 1024.0) as u32,
                storage_gib: offer.disk_space.unwrap_or(0.0) as u32,
                price_cents_per_hour: (total_price * 100.0) as u32,
                regions: vec![offer.geolocation.unwrap_or_else(|| "unknown".to_string())],
            });
        }

        Ok(types)
    }

    fn launch(&self, opts: &LaunchOptions) -> Result<Vec<String>> {
        let search_body = serde_json::json!({
            "verified": {"eq": true},
            "external": {"eq": false},
            "rentable": {"eq": true},
            "gpu_name": {"eq": opts.instance_type},
            "num_gpus": {"gte": 1},
            "type": "on-demand",
            "order": [["dph_total", "asc"]],
            "limit": 1,
        });

        let search_response: BundlesResponse = self.post("/bundles/", &search_body)?;

        let offer = search_response
            .offers
            .first()
            .ok_or_else(|| anyhow!("No offers found for {}", opts.instance_type))?;

        let mut create_body = serde_json::json!({
            "client_id": "me",
            "image": "pytorch/pytorch:latest",
            "disk": 50,
            "onstart": "",
        });

        if let Some(name) = &opts.name {
            create_body["label"] = serde_json::json!(name);
        }

        let create_response: CreateResponse =
            self.put(&format!("/asks/{}/", offer.id), &create_body)?;

        let contract_id = create_response
            .new_contract
            .ok_or_else(|| anyhow!("No contract returned from Vast.ai"))?;

        Ok(vec![contract_id.to_string()])
    }

    fn terminate(&self, ids: &[String]) -> Result<()> {
        for id in ids {
            self.delete(&format!("/instances/{}/", id))?;
        }
        Ok(())
    }
}
