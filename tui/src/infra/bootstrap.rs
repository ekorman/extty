pub fn generate_script(
    python_version: &str,
    command: &str,
    skip_tmux: bool,
    project_dir: &str,
    git_hash: Option<&str>,
    run_command: &str,
) -> String {
    let skip_tmux_str = if skip_tmux { "true" } else { "false" };
    let command_escaped = command.replace('\\', "\\\\").replace('"', "\\\"");

    let mut env_exports = String::new();
    if let Some(hash) = git_hash {
        env_exports.push_str(&format!("export EXTTY_GIT_HASH=\"{}\"\n", hash));
    }
    let run_command_escaped = run_command.replace('\\', "\\\\").replace('"', "\\\"");
    env_exports.push_str(&format!(
        "export EXTTY_RUN_COMMAND=\"{}\"",
        run_command_escaped
    ));

    format!(
        r##"#!/bin/bash

# Ensure common paths are available in non-interactive SSH sessions
export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"

# Forwarded from host by extty run
{env_exports}

# Disable provider auto-tmux (Vast.ai)
touch ~/.no_auto_tmux

# tmux handling
if [ -z "$TMUX" ] && [ "{skip_tmux}" != "true" ]; then
    if [ "$1" != "--in-tmux" ]; then
        exec tmux new-session -s extty "$0 --in-tmux"
    fi
fi

# Install build tools (needed for torch.compile)
if ! command -v gcc &> /dev/null; then
    echo "Installing build-essential..."
    sudo apt update && sudo apt install -y build-essential
fi

# Fix missing libcuda.so symlink (needed for torch.compile on GPU VMs)
for dir in /lib/x86_64-linux-gnu /usr/lib/x86_64-linux-gnu; do
    if [ -f "$dir/libcuda.so.1" ] && [ ! -e "$dir/libcuda.so" ]; then
        sudo ln -sf "$dir/libcuda.so.1" "$dir/libcuda.so"
    fi
done
if [ -f /lib/x86_64-linux-gnu/libcuda.so ] || [ -f /usr/lib/x86_64-linux-gnu/libcuda.so ]; then
    sudo ldconfig
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
cd ~/"{project_dir}"

# Install extty GPU dependencies if NVIDIA GPU is present
if command -v nvidia-smi &> /dev/null && [ -f pyproject.toml ]; then
    echo "NVIDIA GPU detected — installing extty[gpu]..."
    uv pip install --quiet extty[gpu]
fi

# Run command if specified
if [ -n "{command}" ]; then
    echo "Running: {command}"
    {command}
fi

# Drop into a shell so the session stays alive
exec $SHELL
"##,
        skip_tmux = skip_tmux_str,
        python_version = python_version,
        command = command_escaped,
        project_dir = project_dir,
        env_exports = env_exports
    )
}

pub fn project_remote_dir(local_path: &std::path::Path) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let name = local_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("project");

    let canonical = local_path
        .canonicalize()
        .unwrap_or_else(|_| local_path.to_path_buf());
    let mut hasher = DefaultHasher::new();
    canonical.hash(&mut hasher);
    let hash = format!("{:x}", hasher.finish());
    let short_hash = &hash[..6];

    format!("extty-projects/{}-{}", name, short_hash)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_script_basic() {
        let script = generate_script(
            "3.11",
            "uv run train.py",
            false,
            "extty-projects/myproj",
            Some("abc123def456"),
            "extty run uv run train.py",
        );
        assert!(script.contains("uv python install 3.11"));
        assert!(script.contains("uv run train.py"));
        assert!(script.contains("cd ~/\"extty-projects/myproj\""));
        assert!(script.contains("exec $SHELL"));
        assert!(script.contains("touch ~/.no_auto_tmux"));
        assert!(script.contains("sudo apt install -y build-essential"));
        assert!(script.contains("libcuda.so"));
        assert!(script.contains("uv pip install --quiet extty[gpu]"));
        assert!(script.contains("export EXTTY_GIT_HASH=\"abc123def456\""));
        assert!(script.contains("export EXTTY_RUN_COMMAND=\"extty run uv run train.py\""));
    }

    #[test]
    fn test_generate_script_skip_tmux() {
        let script = generate_script("3.11", "", true, "extty-projects/test", None, "extty run");
        assert!(script.contains(r#"[ "true" != "true" ]"#));
        assert!(!script.contains("EXTTY_GIT_HASH"));
        assert!(script.contains("export EXTTY_RUN_COMMAND=\"extty run\""));
    }

    #[test]
    fn test_project_remote_dir() {
        let path = std::path::Path::new("/Users/eric/repos/my-project");
        let dir = project_remote_dir(path);
        assert!(dir.starts_with("extty-projects/my-project-"));
        assert_eq!(dir.len(), "extty-projects/my-project-".len() + 6);

        let dir2 = project_remote_dir(path);
        assert_eq!(dir, dir2, "same path should produce same dir");

        let other = std::path::Path::new("/other/path/my-project");
        let dir3 = project_remote_dir(other);
        assert_ne!(dir, dir3, "different paths with same name should differ");
    }
}
