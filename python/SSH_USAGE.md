# SSH Server Support

`extty` now supports reading runs from remote servers via SSH. This allows you to list and view runs stored on remote machines without needing to set up a web server or transfer files manually.

## Installation

To use SSH features, install extty with the SSH extra:

```bash
pip install extty[ssh]
```

This installs the required `paramiko` library for SSH connections.

## Usage

### List runs on a remote server

```bash
extty --server user@host list
```

This will connect to the remote server via SSH and list all runs in `~/.ex/runs/`.

Examples:
```bash
# List runs on a specific server
extty --server alice@example.com list

# If your SSH config has the host configured, you can omit the username
extty --server ml-server list
```

### Show run data from a remote server

```bash
extty --server user@host show <run_name>
```

This will fetch the complete run data (metadata, metrics, system stats) from the remote server and display it as JSON.

Example:
```bash
extty --server alice@example.com show 2024-01-15_14-32-00_a1b2
```

### Push a remote run to a web server

You can also combine SSH with the push functionality to forward remote runs to a web server:

```bash
# Push a specific remote run to your web server
extty --server alice@example.com push my-run --push-server https://myserver.com

# Push all remote runs to your web server
extty --server alice@example.com push --all --push-server https://myserver.com
```

## SSH Configuration

The SSH connection uses your existing SSH configuration:

- **SSH keys**: Automatically uses SSH agent and local keys from `~/.ssh/`
- **known_hosts**: Follows standard SSH host key verification
- **SSH config**: Respects settings in `~/.ssh/config`

For example, if you have this in your `~/.ssh/config`:

```
Host ml-server
    HostName 192.168.1.100
    User alice
    IdentityFile ~/.ssh/ml_key
```

You can simply use:

```bash
extty --server ml-server list
```

## Remote Directory

By default, `extty` looks for runs in `~/.ex/runs/` on the remote server. This matches the default local directory structure.

## Security

- Uses SSH for secure, encrypted connections
- Leverages your existing SSH credentials (keys, agent)
- No credentials are stored or transmitted except through standard SSH
- Follows SSH best practices (host key verification, key-based auth)

## Troubleshooting

### Connection Refused

If you get "Connection refused" errors:
1. Verify you can connect manually: `ssh user@host`
2. Check that the remote server has SSH enabled
3. Ensure your SSH keys are properly configured

### No Runs Found

If no runs are found on the remote server:
1. Verify runs exist: `ssh user@host ls ~/.ex/runs/`
2. Check that runs have `meta.json` files
3. Ensure the remote user has read permissions

### Import Error

If you get "paramiko is required" error:
- Install the SSH extra: `pip install extty[ssh]`
- Or install paramiko directly: `pip install paramiko>=3.0.0`
