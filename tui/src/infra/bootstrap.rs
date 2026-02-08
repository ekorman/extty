pub struct BootstrapOptions {
    pub python_version: String,
    pub command: String,
    pub skip_tmux: bool,
    pub instance_id: Option<String>,
    pub provider: Option<String>,
    pub auto_shutdown_delay: Option<u64>,
}

#[cfg(test)]
pub fn generate_script(python_version: &str, command: &str, skip_tmux: bool) -> String {
    generate_script_with_options(&BootstrapOptions {
        python_version: python_version.to_string(),
        command: command.to_string(),
        skip_tmux,
        instance_id: None,
        provider: None,
        auto_shutdown_delay: None,
    })
}

pub fn generate_script_with_options(opts: &BootstrapOptions) -> String {
    let skip_tmux_str = if opts.skip_tmux { "true" } else { "false" };
    let command_escaped = opts.command.replace('\\', "\\\\").replace('"', "\\\"");

    let auto_shutdown_enabled = opts.instance_id.is_some() && opts.auto_shutdown_delay.is_some();

    let env_vars = if let (Some(instance_id), Some(provider), Some(delay)) = (
        &opts.instance_id,
        &opts.provider,
        &opts.auto_shutdown_delay,
    ) {
        format!(
            r#"
export EXTTY_INSTANCE_ID="{instance_id}"
export EXTTY_PROVIDER="{provider}"
export EXTTY_AUTO_SHUTDOWN_DELAY="{delay}"
"#,
        )
    } else {
        String::new()
    };

    let tail = if auto_shutdown_enabled {
        String::new()
    } else {
        "\n# Drop into a shell so the session stays alive\nexec $SHELL\n".to_string()
    };

    format!(
        r##"#!/bin/bash
set -e

# Disable provider auto-tmux (Vast.ai)
touch ~/.no_auto_tmux

# tmux handling
if [ -z "$TMUX" ] && [ "{skip_tmux}" != "true" ]; then
    if [ "$1" != "--in-tmux" ]; then
        exec tmux new-session -s extty "$0 --in-tmux"
    fi
fi

# Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Ensure uv is in PATH
export PATH="$HOME/.local/bin:$PATH"

# Install Python version
echo "Installing Python {python_version}..."
uv python install {python_version}

# Code already synced via rsync before SSH
cd ~/project
{env_vars}
# Run command if specified
if [ -n "{command}" ]; then
    echo "Running: {command}"
    {command}
fi
{tail}"##,
        skip_tmux = skip_tmux_str,
        python_version = opts.python_version,
        command = command_escaped,
        env_vars = env_vars,
        tail = tail,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_script_basic() {
        let script = generate_script("3.11", "uv run train.py", false);
        assert!(script.contains("uv python install 3.11"));
        assert!(script.contains("uv run train.py"));
        assert!(script.contains("cd ~/project"));
        assert!(script.contains("exec $SHELL"));
        assert!(script.contains("touch ~/.no_auto_tmux"));
    }

    #[test]
    fn test_generate_script_skip_tmux() {
        let script = generate_script("3.11", "", true);
        assert!(script.contains(r#"[ "true" != "true" ]"#));
    }

    #[test]
    fn test_generate_script_with_auto_shutdown() {
        let script = generate_script_with_options(&BootstrapOptions {
            python_version: "3.12".to_string(),
            command: "uv run train.py".to_string(),
            skip_tmux: false,
            instance_id: Some("abc123".to_string()),
            provider: Some("lambda".to_string()),
            auto_shutdown_delay: Some(10),
        });
        assert!(script.contains("EXTTY_INSTANCE_ID=\"abc123\""));
        assert!(script.contains("EXTTY_PROVIDER=\"lambda\""));
        assert!(script.contains("EXTTY_AUTO_SHUTDOWN_DELAY=\"10\""));
        assert!(!script.contains("exec $SHELL"));
    }

    #[test]
    fn test_generate_script_without_auto_shutdown() {
        let script = generate_script_with_options(&BootstrapOptions {
            python_version: "3.12".to_string(),
            command: "uv run train.py".to_string(),
            skip_tmux: false,
            instance_id: None,
            provider: None,
            auto_shutdown_delay: None,
        });
        assert!(!script.contains("EXTTY_INSTANCE_ID"));
        assert!(script.contains("exec $SHELL"));
    }
}
