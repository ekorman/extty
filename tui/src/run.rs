use std::fs;
use std::io::Write;
use std::os::unix::process::CommandExt;
use std::path::PathBuf;
use std::process::Command;
use std::thread;
use std::time::Duration;

use anyhow::{Result, bail};
use chrono::Local;
use dialoguer::{Select, theme::ColorfulTheme};

use crate::infra::{
    self, InfraConfig, Instance, InstanceStatus, Provider, generate_script, get_provider,
    project_remote_dir,
};

pub struct RunOptions {
    pub provider: Option<String>,
    pub instance_id: Option<String>,
    pub python_version: String,
    pub skip_tmux: bool,
    pub exclude: Vec<String>,
    pub command: Vec<String>,
}

struct LogFile {
    path: PathBuf,
    file: fs::File,
}

impl LogFile {
    fn create() -> Result<Self> {
        let dir = dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".extty")
            .join("logs");
        fs::create_dir_all(&dir)?;

        let timestamp = Local::now().format("%Y%m%d-%H%M%S");
        let path = dir.join(format!("launch-{}.log", timestamp));
        let file = fs::File::create(&path)?;
        Ok(Self { path, file })
    }

    fn log(&mut self, msg: &str) {
        let ts = Local::now().format("[%Y-%m-%d %H:%M:%S%.3f]");
        let line = format!("{} {}\n", ts, msg);
        let _ = self.file.write_all(line.as_bytes());
    }
}

pub fn run(opts: RunOptions) -> Result<()> {
    let config = infra::load_config()?;
    let mut log = LogFile::create()?;
    log.log("Launch started");

    let instance = select_instance(&config, &opts, &mut log)?;
    let ip = instance
        .ip
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("Instance has no IP address"))?;

    log.log(&format!(
        "Selected instance: {} ({}) [{}]",
        instance.display_name(),
        ip,
        instance.provider.display_name()
    ));

    let (host, port) = parse_host_port(ip);
    let ssh_user = &instance.ssh_user;

    let ssh_port_args = port.as_deref().map(|p| format!("-p {}", p)).unwrap_or_default();
    let ssh_base = if ssh_port_args.is_empty() {
        format!("ssh -o StrictHostKeyChecking=no {}@{}", ssh_user, host)
    } else {
        format!("ssh -o StrictHostKeyChecking=no {} {}@{}", ssh_port_args, ssh_user, host)
    };

    let cwd = std::env::current_dir()?;
    let remote_dir = project_remote_dir(&cwd);

    log.log(&format!("Creating remote directory: {}", remote_dir));
    let mkdir_status = Command::new("ssh")
        .arg("-o").arg("StrictHostKeyChecking=no")
        .args(port.as_deref().map(|p| vec!["-p", p]).unwrap_or_default())
        .arg(format!("{}@{}", ssh_user, host))
        .arg(format!("mkdir -p $HOME/{}", remote_dir))
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped())
        .output();
    match mkdir_status {
        Ok(output) if !output.status.success() => {
            let stderr = String::from_utf8_lossy(&output.stderr);
            log.log(&format!("mkdir failed: {}", stderr.trim()));
            bail!("Failed to create remote directory: {}", stderr.trim());
        }
        Err(e) => bail!("Failed to run ssh for mkdir: {}", e),
        _ => {}
    }

    rsync_cwd(&cwd, &remote_dir, &host, port.as_deref(), ssh_user, &opts.exclude, &mut log)?;
    copy_s3_config(&ssh_base, &host, port.as_deref(), ssh_user, &mut log);
    upload_bootstrap_script(&ssh_base, &opts, &remote_dir, &mut log)?;

    log.log("Exec into SSH session");
    println!("Connecting to {}...", instance.display_name());

    let mut cmd = Command::new("ssh");
    cmd.arg("-t").arg("-o").arg("StrictHostKeyChecking=no");
    if let Some(p) = &port {
        cmd.arg("-p").arg(p);
    }
    cmd.arg(format!("{}@{}", ssh_user, host))
        .arg("bash")
        .arg("/tmp/extty_bootstrap.sh");

    let err = cmd.exec();
    bail!("Failed to exec ssh: {}", err);
}

fn parse_host_port(ip: &str) -> (String, Option<String>) {
    if ip.contains(':') {
        let parts: Vec<&str> = ip.split(':').collect();
        (parts[0].to_string(), Some(parts[1].to_string()))
    } else {
        (ip.to_string(), None)
    }
}

fn resolve_provider(name: &str) -> Result<Provider> {
    match name.to_lowercase().as_str() {
        "lambda" => Ok(Provider::Lambda),
        "vast" => Ok(Provider::Vast),
        "prime" => Ok(Provider::Prime),
        "local" => Ok(Provider::Local),
        _ => bail!("Unknown provider: {}. Valid: lambda, vast, prime, local", name),
    }
}

fn fetch_running_instances(
    config: &InfraConfig,
    provider: Provider,
    log: &mut LogFile,
) -> Vec<Instance> {
    if provider == Provider::Local {
        let instances: Vec<Instance> = config.local.iter().map(|m| m.to_instance()).collect();
        log.log(&format!("Local: {} machine(s)", instances.len()));
        return instances;
    }

    let provider_config = config.get_provider_config(provider);
    let api_key = match provider_config.api_key {
        Some(ref k) => k.as_str(),
        None => {
            log.log(&format!("No API key for {}, skipping", provider.display_name()));
            return vec![];
        }
    };

    let cloud = get_provider(provider, api_key);
    match cloud.list_instances() {
        Ok(instances) => {
            let running: Vec<Instance> = instances
                .into_iter()
                .filter(|i| i.status == InstanceStatus::Running)
                .collect();
            log.log(&format!(
                "{}: {} running instance(s)",
                provider.display_name(),
                running.len()
            ));
            running
        }
        Err(e) => {
            log.log(&format!("Failed to list {} instances: {}", provider.display_name(), e));
            vec![]
        }
    }
}

fn select_instance(
    config: &InfraConfig,
    opts: &RunOptions,
    log: &mut LogFile,
) -> Result<Instance> {
    if let Some(ref id_or_name) = opts.instance_id {
        return find_instance_by_id_or_name(config, id_or_name, opts, log);
    }

    let providers_to_check: Vec<Provider> = if let Some(ref name) = opts.provider {
        vec![resolve_provider(name)?]
    } else {
        vec![config.default_provider]
    };

    let mut instances: Vec<Instance> = providers_to_check
        .iter()
        .flat_map(|p| fetch_running_instances(config, *p, log))
        .collect();

    if instances.is_empty() && opts.provider.is_none() {
        log.log("No running instances on default provider, trying all providers");
        instances = Provider::all()
            .iter()
            .filter(|p| **p != config.default_provider)
            .flat_map(|p| fetch_running_instances(config, *p, log))
            .collect();
    }

    if instances.is_empty() {
        bail!("No running instances found");
    }

    let labels: Vec<String> = instances
        .iter()
        .map(|i| {
            format!(
                "{} - {} ({}) [{}]",
                i.display_name(),
                i.instance_type,
                i.ip.as_deref().unwrap_or("?"),
                i.provider.display_name()
            )
        })
        .collect();

    let theme = ColorfulTheme::default();
    let selection = Select::with_theme(&theme)
        .with_prompt("Select instance")
        .items(&labels)
        .default(0)
        .interact()?;

    Ok(instances.remove(selection))
}

fn find_instance_by_id_or_name(
    config: &InfraConfig,
    id_or_name: &str,
    opts: &RunOptions,
    log: &mut LogFile,
) -> Result<Instance> {
    let providers_to_check: Vec<Provider> = if let Some(ref name) = opts.provider {
        vec![resolve_provider(name)?]
    } else {
        Provider::all().to_vec()
    };

    for provider in &providers_to_check {
        let instances = fetch_running_instances(config, *provider, log);
        if let Some(inst) = instances.into_iter().find(|i| {
            i.id == id_or_name || i.name.as_deref() == Some(id_or_name)
        }) {
            return Ok(inst);
        }
    }

    bail!("Instance '{}' not found or not running", id_or_name)
}

fn rsync_cwd(
    cwd: &std::path::Path,
    remote_dir: &str,
    host: &str,
    port: Option<&str>,
    ssh_user: &str,
    extra_excludes: &[String],
    log: &mut LogFile,
) -> Result<()> {
    let src = format!("{}/", cwd.display());
    let dest = format!("{}@{}:{}/", ssh_user, host, remote_dir);

    let exttyignore_path = cwd.join(".exttyignore");

    let delays = [1, 2, 4];
    for (attempt, delay) in delays.iter().enumerate() {
        let attempt_num = attempt + 1;
        log.log(&format!("rsync attempt {}/3", attempt_num));
        println!("Syncing code (attempt {}/3)...", attempt_num);

        let mut rsync_cmd = Command::new("rsync");
        rsync_cmd.args(["-az", "--progress"]);

        if exttyignore_path.exists() {
            let filter_arg = format!("merge {}", exttyignore_path.display());
            rsync_cmd.args(["--filter", &filter_arg]);
        }

        rsync_cmd.args([
            "--exclude", ".git",
            "--exclude", "__pycache__",
            "--exclude", ".venv",
            "--exclude", "*.pyc",
            "--exclude", ".mypy_cache",
            "--exclude", "*.egg-info",
        ]);

        for pattern in extra_excludes {
            rsync_cmd.args(["--exclude", pattern]);
        }

        let ssh_arg = match port {
            Some(p) => format!("ssh -o StrictHostKeyChecking=no -p {}", p),
            None => "ssh -o StrictHostKeyChecking=no".to_string(),
        };
        rsync_cmd.args(["-e", &ssh_arg]);
        rsync_cmd.arg(&src).arg(&dest);
        rsync_cmd
            .stdout(std::process::Stdio::inherit())
            .stderr(std::process::Stdio::piped());

        match rsync_cmd.spawn().and_then(|child| child.wait_with_output()) {
            Ok(output) => {
                let stderr = String::from_utf8_lossy(&output.stderr);
                if !stderr.is_empty() {
                    log.log(&format!("rsync stderr: {}", stderr.trim()));
                    eprint!("{}", stderr);
                }

                if output.status.success() {
                    println!("Code synced successfully.");
                    log.log("rsync succeeded");
                    return Ok(());
                }

                log.log(&format!("rsync failed with exit code: {}", output.status));
            }
            Err(e) => {
                log.log(&format!("rsync spawn error: {}", e));
            }
        }

        if attempt_num < 3 {
            log.log(&format!("Retrying in {}s...", delay));
            println!("rsync failed, retrying in {}s...", delay);
            thread::sleep(Duration::from_secs(*delay));
        }
    }

    bail!(
        "rsync failed after 3 attempts. Check log: {}",
        log.path.display()
    )
}

fn copy_s3_config(
    ssh_base: &str,
    host: &str,
    port: Option<&str>,
    ssh_user: &str,
    log: &mut LogFile,
) {
    let s3_config_path = dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".extty")
        .join("s3")
        .join("config.toml");

    if !s3_config_path.exists() {
        return;
    }

    log.log("Copying S3 config...");
    println!("Copying S3 config...");

    let scp_port_arg = port.map(|p| format!("-P {}", p)).unwrap_or_default();

    let result = Command::new("bash")
        .arg("-c")
        .arg(format!(
            "{} 'mkdir -p ~/.extty/s3' && scp -o StrictHostKeyChecking=no {} {} {}@{}:~/.extty/s3/config.toml",
            ssh_base,
            scp_port_arg,
            s3_config_path.display(),
            ssh_user,
            host,
        ))
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped())
        .output();

    match result {
        Ok(output) if output.status.success() => {
            log.log("S3 config copied");
        }
        Ok(output) => {
            let stderr = String::from_utf8_lossy(&output.stderr);
            log.log(&format!("S3 config copy failed: {}", stderr.trim()));
            eprintln!("Warning: failed to copy S3 config (non-fatal)");
        }
        Err(e) => {
            log.log(&format!("S3 config copy error: {}", e));
            eprintln!("Warning: failed to copy S3 config (non-fatal)");
        }
    }
}

fn upload_bootstrap_script(
    ssh_base: &str,
    opts: &RunOptions,
    remote_dir: &str,
    log: &mut LogFile,
) -> Result<()> {
    log.log("Uploading bootstrap script...");
    println!("Uploading bootstrap script...");

    let command_str = opts.command.join(" ");
    let script = generate_script(&opts.python_version, &command_str, opts.skip_tmux, remote_dir);

    let mut child = Command::new("bash")
        .arg("-c")
        .arg(format!(
            "{} 'cat > /tmp/extty_bootstrap.sh && chmod +x /tmp/extty_bootstrap.sh'",
            ssh_base
        ))
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped())
        .spawn()?;

    if let Some(ref mut stdin) = child.stdin {
        stdin.write_all(script.as_bytes())?;
    }
    drop(child.stdin.take());

    let output = child.wait_with_output()?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        log.log(&format!("Script upload failed: {}", stderr.trim()));
        bail!(
            "Bootstrap script upload failed. Check log: {}",
            log.path.display()
        );
    }

    log.log("Bootstrap script uploaded");
    Ok(())
}
