use std::collections::HashMap;

use anyhow::{Context, Result, anyhow};
use reqwest::blocking::Client;
use reqwest::header::{AUTHORIZATION, HeaderMap, HeaderValue};
use serde::Deserialize;

use crate::infra::models::{Instance, InstanceStatus, InstanceType, LaunchOptions, Provider};
use crate::infra::providers::CloudProvider;

const BASE_URL: &str = "https://console.vast.ai/api/v0";
const DISK_GB: u32 = 30;

fn parse_instance_type(instance_type: &str) -> Result<(&str, u32)> {
    if let Some(pos) = instance_type.rfind('x') {
        let (gpu_name, suffix) = instance_type.split_at(pos);
        if let Ok(num) = suffix[1..].parse::<u32>() {
            return Ok((gpu_name, num));
        }
    }
    Err(anyhow!("Invalid instance type format: {}", instance_type))
}

fn normalize_status(raw: &str) -> InstanceStatus {
    match raw.to_lowercase().as_str() {
        "running" => InstanceStatus::Running,
        "loading" => InstanceStatus::Booting,
        "created" | "scheduling" => InstanceStatus::Pending,
        "exited" => InstanceStatus::Stopped,
        "destroying" => InstanceStatus::Stopping,
        "destroyed" => InstanceStatus::Terminated,
        "offline" | "error" => InstanceStatus::Error,
        _ => InstanceStatus::Pending,
    }
}

fn direct_address(instance: &VastInstance) -> Option<String> {
    let host = instance.public_ipaddr.as_deref()?;
    let port = instance
        .ports
        .as_ref()?
        .get("22/tcp")?
        .iter()
        .find_map(|binding| binding.host_port.as_deref())?;
    Some(format!("{}:{}", host, port))
}

/// Vast's SSH proxy reserves `ssh_port` for the host machine's sshd and forwards
/// `ssh_port + 1` into the container whenever the image exposes Jupyter, which is
/// why the raw `ssh_port` field is off by one from the port the console shows.
fn proxy_address(instance: &VastInstance) -> Option<String> {
    let host = instance.ssh_host.as_deref()?;
    let base = instance.ssh_port?;
    let port = match instance.image_runtype.as_deref() {
        Some(runtype) if runtype.contains("jupyter") => base.saturating_add(1),
        _ => base,
    };
    Some(format!("{}:{}", host, port))
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
    public_ipaddr: Option<String>,
    #[serde(default)]
    ports: Option<HashMap<String, Vec<PortBinding>>>,
    image_runtype: Option<String>,
    actual_status: Option<String>,
    status_msg: Option<String>,
    gpu_name: Option<String>,
    geolocation: Option<String>,
    dph_total: Option<f64>,
}

#[derive(Deserialize)]
struct PortBinding {
    #[serde(rename = "HostPort")]
    host_port: Option<String>,
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
    gpu_ram: Option<u64>,
    cpu_cores_effective: Option<f64>,
    cpu_ram: Option<f64>,
    disk_space: Option<f64>,
    dph_base: Option<f64>,
    dph_total: Option<f64>,
    geolocation: Option<String>,
}

#[derive(Deserialize)]
struct CreateResponse {
    new_contract: Option<i64>,
}

impl CloudProvider for VastProvider {
    fn ssh_user(&self) -> &str {
        "root"
    }

    fn list_instances(&self) -> Result<Vec<Instance>> {
        let response: InstancesResponse = self.get("/instances/")?;

        Ok(response
            .instances
            .into_iter()
            .map(|i| {
                let ip = direct_address(&i).or_else(|| proxy_address(&i));
                let raw_status = i
                    .actual_status
                    .or(i.status_msg)
                    .unwrap_or_else(|| "unknown".to_string());

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
                    price_cents_per_hour: i.dph_total.map(|d| (d * 100.0) as u32),
                }
            })
            .collect())
    }

    fn list_instance_types(&self) -> Result<Vec<InstanceType>> {
        let body = serde_json::json!({
            "verified": {"eq": true},
            "rentable": {"eq": true},
            "num_gpus": {"gte": 1},
            "type": "on-demand",
            "order": [["dph_total", "asc"]],
            "limit": 1000,
        });

        let response: BundlesResponse = self.post("/bundles/", &body)?;

        let mut grouped: HashMap<String, InstanceType> = HashMap::new();
        let mut insertion_order: Vec<String> = Vec::new();

        for offer in response.offers {
            let gpu_name = offer.gpu_name.unwrap_or_else(|| "unknown".to_string());
            let num_gpus = offer.num_gpus.unwrap_or(1);
            let type_key = format!("{}x{}", gpu_name, num_gpus);
            let region = offer.geolocation.unwrap_or_else(|| "unknown".to_string());
            let total_price = offer
                .dph_total
                .unwrap_or_else(|| offer.dph_base.unwrap_or(0.0) * num_gpus as f64);
            let price_cents = (total_price * 100.0) as u32;

            if let Some(existing) = grouped.get_mut(&type_key) {
                if !existing.regions.contains(&region) {
                    existing.regions.push(region.clone());
                }
                let price_key = format!("price:{}", region);
                existing
                    .metadata
                    .entry(price_key)
                    .or_insert_with(|| price_cents.to_string());
            } else {
                let gpu_ram_gb = offer.gpu_ram.map(|r| r / 1024);
                let gpu_description = gpu_ram_gb.map(|gb| format!("{}GB", gb));

                let mut metadata = HashMap::new();
                metadata.insert(format!("price:{}", region), price_cents.to_string());

                insertion_order.push(type_key.clone());
                grouped.insert(
                    type_key.clone(),
                    InstanceType {
                        name: type_key,
                        description: Some(format!("{}x {}", num_gpus, gpu_name)),
                        gpu_count: num_gpus,
                        gpu_name: Some(gpu_name),
                        gpu_description,
                        gpu_memory_gib: gpu_ram_gb.unwrap_or(0) as u32,
                        vcpus: offer.cpu_cores_effective.unwrap_or(0.0) as u32,
                        memory_gib: (offer.cpu_ram.unwrap_or(0.0) / 1024.0) as u32,
                        storage_gib: offer.disk_space.unwrap_or(0.0) as u32,
                        price_cents_per_hour: price_cents,
                        regions: vec![region],
                        metadata,
                    },
                );
            }
        }

        Ok(insertion_order
            .into_iter()
            .filter_map(|key| grouped.remove(&key))
            .collect())
    }

    fn launch(&self, opts: &LaunchOptions) -> Result<Vec<String>> {
        let (gpu_name, num_gpus) = parse_instance_type(&opts.instance_type)?;

        let search_body = serde_json::json!({
            "verified": {"eq": true},
            "rentable": {"eq": true},
            "gpu_name": {"eq": gpu_name},
            "num_gpus": {"eq": num_gpus},
            "disk_space": {"gte": DISK_GB},
            "type": "on-demand",
            "order": [["dph_total", "asc"]],
            "limit": 100,
        });

        let search_response: BundlesResponse = self.post("/bundles/", &search_body)?;

        let offer = if let Some(region) = &opts.region {
            search_response
                .offers
                .iter()
                .find(|o| o.geolocation.as_deref() == Some(region.as_str()))
                .ok_or_else(|| {
                    anyhow!(
                        "No offers found for {} in {} — the offer may have been taken",
                        opts.instance_type,
                        region
                    )
                })?
        } else {
            search_response
                .offers
                .first()
                .ok_or_else(|| anyhow!("No offers found for {}", opts.instance_type))?
        };

        let mut create_body = serde_json::json!({
            "client_id": "me",
            "image": "pytorch/pytorch:latest",
            "disk": DISK_GB,
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

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(json: &str) -> VastInstance {
        serde_json::from_str(json).expect("valid instance json")
    }

    #[test]
    fn test_direct_address_prefers_mapped_ssh_port() {
        let instance = parse(
            r#"{
                "id": 49019841,
                "ssh_host": "ssh6.vast.ai",
                "ssh_port": 19840,
                "public_ipaddr": "151.237.25.16",
                "image_runtype": "jupyter_direc ssh_direc ssh_proxy",
                "ports": {
                    "8080/tcp": [{"HostIp": "0.0.0.0", "HostPort": "29061"}],
                    "22/tcp": [{"HostIp": "0.0.0.0", "HostPort": "29283"}, {"HostIp": "::", "HostPort": "29283"}]
                }
            }"#,
        );
        assert_eq!(
            direct_address(&instance).as_deref(),
            Some("151.237.25.16:29283")
        );
    }

    #[test]
    fn test_proxy_address_offsets_jupyter_images() {
        let instance = parse(
            r#"{
                "id": 49019841,
                "ssh_host": "ssh6.vast.ai",
                "ssh_port": 19840,
                "image_runtype": "jupyter_direc ssh_direc ssh_proxy",
                "ports": null
            }"#,
        );
        assert_eq!(direct_address(&instance), None);
        assert_eq!(
            proxy_address(&instance).as_deref(),
            Some("ssh6.vast.ai:19841")
        );
    }

    #[test]
    fn test_proxy_address_without_jupyter_uses_raw_port() {
        let instance = parse(
            r#"{
                "id": 1,
                "ssh_host": "ssh3.vast.ai",
                "ssh_port": 19880,
                "image_runtype": "ssh_direc ssh_proxy"
            }"#,
        );
        assert_eq!(
            proxy_address(&instance).as_deref(),
            Some("ssh3.vast.ai:19880")
        );
    }

    #[test]
    fn test_no_address_without_host() {
        let instance = parse(r#"{"id": 1}"#);
        assert_eq!(direct_address(&instance), None);
        assert_eq!(proxy_address(&instance), None);
    }
}
