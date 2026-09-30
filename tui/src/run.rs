use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::thread;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use chrono::Local;
use dialoguer::{Select, theme::ColorfulTheme};

use crate::data::{self, RunStatus};
use crate::infra::{
    self, InfraConfig, Instance, InstanceStatus, Provider, ScriptOptions, generate_script,
    get_provider, project_remote_dir,
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
        let dir = crate::paths::extty_home().join("logs");
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

    let ssh_port_args = port
        .as_deref()
        .map(|p| format!("-p {}", p))
        .unwrap_or_default();
    let ssh_base = if ssh_port_args.is_empty() {
        format!("ssh -o StrictHostKeyChecking=no {}@{}", ssh_user, host)
    } else {
        format!(
            "ssh -o StrictHostKeyChecking=no {} {}@{}",
            ssh_port_args, ssh_user, host
        )
    };

    let cwd = std::env::current_dir()?;
    let remote_dir = project_remote_dir(&cwd);

    let git_hash = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(&cwd)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .output()
        .ok()
        .and_then(|output| {
            if output.status.success() {
                String::from_utf8(output.stdout)
                    .ok()
                    .map(|s| s.trim().to_string())
            } else {
                None
            }
        });
    let run_command: String = std::env::args().collect::<Vec<_>>().join(" ");

    log.log(&format!("Creating remote directory: {}", remote_dir));
    let mkdir_status = Command::new("ssh")
        .arg("-o")
        .arg("StrictHostKeyChecking=no")
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

    rsync_cwd(
        &cwd,
        &remote_dir,
        &host,
        port.as_deref(),
        ssh_user,
        &opts.exclude,
        &mut log,
    )?;
    copy_s3_config(&ssh_base, &mut log)?;
    let command_str = opts.command.join(" ");
    let script_opts = ScriptOptions {
        python_version: &opts.python_version,
        command: &command_str,
        skip_tmux: opts.skip_tmux,
        project_dir: &remote_dir,
        git_hash: git_hash.as_deref(),
        run_command: &run_command,
        instance_id: &instance.id,
        provider: instance.provider.as_str(),
    };
    upload_bootstrap_script(&ssh_base, &script_opts, &mut log)?;

    log.log("Launching SSH session");
    println!("Connecting to {}...", instance.display_name());

    let mut cmd = Command::new("ssh");
    cmd.arg("-t").arg("-o").arg("StrictHostKeyChecking=no");
    if let Some(p) = &port {
        cmd.arg("-p").arg(p);
    }
    cmd.arg(format!("{}@{}", ssh_user, host))
        .arg("bash")
        .arg("/tmp/extty_bootstrap.sh");

    let status = cmd
        .stdin(std::process::Stdio::inherit())
        .stdout(std::process::Stdio::inherit())
        .stderr(std::process::Stdio::inherit())
        .status()
        .context("Failed to launch ssh")?;

    // Restore terminal state after SSH exits — SSH with -t (PTY) and
    // remote tmux can leave the terminal with cursor hidden / raw mode.
    let _ = Command::new("stty").arg("sane").status();
    print!("\x1b[?25h");
    let _ = std::io::stdout().flush();

    log.log(&format!("SSH exited with status: {}", status));
    if !status.success() {
        bail!("SSH exited with status: {}", status);
    }

    Ok(())
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
        _ => bail!(
            "Unknown provider: {}. Valid: lambda, vast, prime, local",
            name
        ),
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
            log.log(&format!(
                "No API key for {}, skipping",
                provider.display_name()
            ));
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
            log.log(&format!(
                "Failed to list {} instances: {}",
                provider.display_name(),
                e
            ));
            vec![]
        }
    }
}

fn count_active_runs_per_instance() -> HashMap<String, usize> {
    let runs = data::load_runs_lightweight();
    let mut counts: HashMap<String, usize> = HashMap::new();
    for run in &runs {
        if run.status != RunStatus::Running {
            continue;
        }
        let instance_id = run
            .config
            .as_ref()
            .and_then(|c| c.get("_instance_id"))
            .and_then(|v| v.as_str());
        if let Some(id) = instance_id {
            *counts.entry(id.to_string()).or_default() += 1;
        }
    }
    counts
}

fn select_instance(config: &InfraConfig, opts: &RunOptions, log: &mut LogFile) -> Result<Instance> {
    if let Some(ref id_or_name) = opts.instance_id {
        return find_instance_by_id_or_name(config, id_or_name, opts, log);
    }

    let providers_to_check: Vec<Provider> = if let Some(ref name) = opts.provider {
        vec![resolve_provider(name)?]
    } else {
        Provider::all().to_vec()
    };

    let mut instances: Vec<Instance> = providers_to_check
        .iter()
        .flat_map(|p| fetch_running_instances(config, *p, log))
        .collect();

    if instances.is_empty() {
        bail!("No running instances found");
    }

    let active_runs = count_active_runs_per_instance();

    let labels: Vec<String> = instances
        .iter()
        .map(|i| {
            let key = format!("{}:{}", i.provider.as_str(), i.id);
            let run_count = active_runs.get(&key).copied().unwrap_or(0);
            let runs_label = if run_count == 1 { "run" } else { "runs" };
            let color = if run_count == 0 {
                "\x1b[32m"
            } else {
                "\x1b[33m"
            };
            format!(
                "{} - {} ({}) [{}] {}({} active {})\x1b[0m",
                i.display_name(),
                i.instance_type,
                i.ip.as_deref().unwrap_or("?"),
                i.provider.display_name(),
                color,
                run_count,
                runs_label,
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
        if let Some(inst) = instances
            .into_iter()
            .find(|i| i.id == id_or_name || i.name.as_deref() == Some(id_or_name))
        {
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
            "--exclude",
            ".git",
            "--exclude",
            "__pycache__",
            "--exclude",
            ".venv",
            "--exclude",
            "*.pyc",
            "--exclude",
            ".mypy_cache",
            "--exclude",
            "*.egg-info",
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

/// Remote command that runs the script on stdin with `sh`, so the install
/// works whatever the remote user's login shell is.
const REMOTE_SH: &str = "sh -s";

const S3_CONFIG_DELIMITER: &str = "EXTTY_S3_CONFIG_EOF";

/// Shell script that installs `config` where the SDK on the remote host looks
/// for it: `$EXTTY_HOME/s3/config.toml`, defaulting to `~/.extty`. The file
/// holds credentials, so it is made owner-only.
fn s3_config_install_script(config: &str) -> Result<String> {
    if config.lines().any(|line| line == S3_CONFIG_DELIMITER) {
        bail!("S3 config contains the line {S3_CONFIG_DELIMITER}");
    }
    let body = if config.ends_with('\n') {
        config.to_string()
    } else {
        format!("{config}\n")
    };
    Ok(format!(
        "set -e\n\
         umask 077\n\
         d=\"${{EXTTY_HOME:-$HOME/.extty}}/s3\"\n\
         mkdir -p \"$d\"\n\
         cat > \"$d/config.toml\" <<'{S3_CONFIG_DELIMITER}'\n\
         {body}{S3_CONFIG_DELIMITER}\n\
         chmod 600 \"$d/config.toml\"\n"
    ))
}

/// Copy the local S3 config to the host reached by `ssh_base`.
///
/// The config is sent over ssh inside an install script rather than with
/// scp: scp's SFTP mode doesn't shell-expand remote paths, so it can't honour
/// a remote `EXTTY_HOME`.
pub fn send_s3_config(ssh_base: &str, config_path: &Path) -> Result<std::process::Output> {
    let config = fs::read_to_string(config_path)
        .with_context(|| format!("Failed to read {}", config_path.display()))?;
    let script = s3_config_install_script(&config)?;
    let mut child = Command::new("bash")
        .arg("-c")
        .arg(format!("{} '{}'", ssh_base, REMOTE_SH))
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .context("Failed to run ssh for S3 config")?;
    if let Some(mut stdin) = child.stdin.take() {
        stdin.write_all(script.as_bytes())?;
    }
    child
        .wait_with_output()
        .context("Failed to run ssh for S3 config")
}

fn copy_s3_config(ssh_base: &str, log: &mut LogFile) -> Result<()> {
    let s3_config_path = crate::s3::config_path();

    if !s3_config_path.exists() {
        return Ok(());
    }

    log.log("Copying S3 config...");
    println!("Copying S3 config...");

    let output = send_s3_config(ssh_base, &s3_config_path)?;

    if output.status.success() {
        log.log("S3 config copied");
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let msg = format!("Failed to copy S3 config: {}", stderr.trim());
        log.log(&msg);
        anyhow::bail!(msg);
    }

    Ok(())
}

fn upload_bootstrap_script(
    ssh_base: &str,
    script_opts: &ScriptOptions,
    log: &mut LogFile,
) -> Result<()> {
    log.log("Uploading bootstrap script...");
    println!("Uploading bootstrap script...");

    let script = generate_script(script_opts);

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

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;
    use std::process::Stdio;

    fn install_config(home: &Path, extty_home: Option<&Path>, config: &str) {
        let mut cmd = Command::new("sh");
        cmd.arg("-s").env("HOME", home).stdin(Stdio::piped());
        match extty_home {
            Some(dir) => cmd.env("EXTTY_HOME", dir),
            None => cmd.env_remove("EXTTY_HOME"),
        };
        let mut child = cmd.spawn().unwrap();
        child
            .stdin
            .take()
            .unwrap()
            .write_all(s3_config_install_script(config).unwrap().as_bytes())
            .unwrap();
        assert!(child.wait().unwrap().success());
    }

    fn install(home: &Path, extty_home: Option<&Path>) {
        install_config(home, extty_home, "bucket = \"b\"\n");
    }

    fn assert_owner_only_config(path: &Path) {
        assert_eq!(fs::read(path).unwrap(), b"bucket = \"b\"\n");
        let mode = fs::metadata(path).unwrap().permissions().mode();
        assert_eq!(mode & 0o777, 0o600);
    }

    #[test]
    fn remote_s3_config_follows_remote_extty_home() {
        let dir = tempfile::tempdir().unwrap();
        let home = dir.path().join("home");
        let scratch = dir.path().join("scratch/extty");

        install(&home, Some(&scratch));

        assert_owner_only_config(&scratch.join("s3/config.toml"));
        assert!(!home.join(".extty").exists());
    }

    #[test]
    fn remote_s3_config_defaults_to_dot_extty() {
        let dir = tempfile::tempdir().unwrap();
        let home = dir.path().join("home");

        install(&home, None);

        assert_owner_only_config(&home.join(".extty/s3/config.toml"));
    }

    /// `sh -c` stands in for ssh: both hand the quoted command to a shell.
    #[test]
    fn send_s3_config_runs_install_through_ssh_quoting() {
        let dir = tempfile::tempdir().unwrap();
        let config = dir.path().join("config.toml");
        fs::write(&config, b"bucket = \"b\"\n").unwrap();
        let scratch = dir.path().join("scratch");

        let ssh_base = format!("EXTTY_HOME={} sh -c", scratch.display());
        let output = send_s3_config(&ssh_base, &config).unwrap();

        assert!(output.status.success(), "{:?}", output);
        assert_owner_only_config(&scratch.join("s3/config.toml"));
    }

    #[test]
    fn remote_s3_config_gets_a_trailing_newline() {
        let dir = tempfile::tempdir().unwrap();
        let home = dir.path().join("home");

        install_config(&home, None, "bucket = \"b\"");

        assert_owner_only_config(&home.join(".extty/s3/config.toml"));
    }

    #[test]
    fn s3_config_containing_the_delimiter_is_refused() {
        let config = format!("bucket = \"b\"\n{S3_CONFIG_DELIMITER}\n");
        assert!(s3_config_install_script(&config).is_err());
    }

    /// Installs a config over real ssh on `EXTTY_SMOKE_SSH_HOST`, under
    /// `EXTTY_SMOKE_REMOTE_DIR`: once through `EXTTY_HOME` and once through the
    /// default `~/.extty`. Run by `scripts/checkpoint_smoke/run.sh`.
    #[test]
    #[ignore]
    fn smoke_send_s3_config_over_ssh() {
        let var = |k: &str| std::env::var(k).unwrap_or_else(|_| panic!("{k} is not set"));
        let host = var("EXTTY_SMOKE_SSH_HOST");
        let remote = var("EXTTY_SMOKE_REMOTE_DIR");
        let dir = tempfile::tempdir().unwrap();
        let config = dir.path().join("config.toml");
        fs::write(&config, b"bucket = \"b\"\n").unwrap();

        for (assignment, installed) in [
            (
                format!("EXTTY_HOME={remote}/extty-home"),
                format!("{remote}/extty-home/s3/config.toml"),
            ),
            (
                format!("HOME={remote}/home"),
                format!("{remote}/home/.extty/s3/config.toml"),
            ),
        ] {
            let ssh_base = format!("ssh -o StrictHostKeyChecking=no {host} {assignment}");
            let output = send_s3_config(&ssh_base, &config).unwrap();
            assert!(output.status.success(), "{:?}", output);

            let check = Command::new("ssh")
                .arg(&host)
                .arg(format!("stat -c %a {installed} && cat {installed}"))
                .output()
                .unwrap();
            assert_eq!(
                String::from_utf8_lossy(&check.stdout),
                "600\nbucket = \"b\"\n",
                "{installed}"
            );
        }
    }
}
