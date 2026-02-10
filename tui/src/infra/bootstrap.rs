pub fn generate_script(
    python_version: &str,
    command: &str,
    skip_tmux: bool,
    project_dir: &str,
) -> String {
    let skip_tmux_str = if skip_tmux { "true" } else { "false" };
    let command_escaped = command.replace('\\', "\\\\").replace('"', "\\\"");

    format!(
        r##"#!/bin/bash
set -e

# Ensure common paths are available in non-interactive SSH sessions
export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"

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
cd ~/{project_dir}

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
        project_dir = project_dir
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
        let script = generate_script("3.11", "uv run train.py", false, "extty-projects/myproj");
        assert!(script.contains("uv python install 3.11"));
        assert!(script.contains("uv run train.py"));
        assert!(script.contains("cd ~/extty-projects/myproj"));
        assert!(script.contains("exec $SHELL"));
        assert!(script.contains("touch ~/.no_auto_tmux"));
    }

    #[test]
    fn test_generate_script_skip_tmux() {
        let script = generate_script("3.11", "", true, "extty-projects/test");
        assert!(script.contains(r#"[ "true" != "true" ]"#));
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
