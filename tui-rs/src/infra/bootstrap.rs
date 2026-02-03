pub fn generate_script(python_version: &str, command: &str, skip_tmux: bool) -> String {
    let skip_tmux_str = if skip_tmux { "true" } else { "false" };
    let command_escaped = command.replace('\\', "\\\\").replace('"', "\\\"");

    format!(
        r##"#!/bin/bash
set -e

# tmux handling (skip if already in tmux or if skip_tmux is set)
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

# Run command if specified
if [ -n "{command}" ]; then
    echo "Running: {command}"
    {command}
else
    echo "Setup complete. Project directory: ~/project"
    exec $SHELL
fi
"##,
        skip_tmux = skip_tmux_str,
        python_version = python_version,
        command = command_escaped
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
    }

    #[test]
    fn test_generate_script_skip_tmux() {
        let script = generate_script("3.11", "", true);
        assert!(script.contains(r#"[ "true" != "true" ]"#));
    }
}
