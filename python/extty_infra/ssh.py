"""SSH and rsync operations via subprocess."""

import subprocess
import sys
import time


def wait_for_ssh(
    host: str, user: str = "ubuntu", timeout: int = 300, interval: int = 5
) -> bool:
    """
    Wait for SSH to become available on a host.

    Parameters
    ----------
    host : str
        The hostname or IP address to connect to.
    user : str
        The SSH user to connect as.
    timeout : int
        Maximum time to wait in seconds.
    interval : int
        Time between connection attempts in seconds.

    Returns
    -------
    bool
        True if SSH is available, False if timeout reached.
    """
    start = time.time()
    while time.time() - start < timeout:
        result = subprocess.run(
            [
                "ssh",
                "-o",
                "ConnectTimeout=5",
                "-o",
                "StrictHostKeyChecking=no",
                "-o",
                "BatchMode=yes",
                f"{user}@{host}",
                "echo ok",
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return True
        time.sleep(interval)
    return False


def run_remote(
    host: str, cmd: str, user: str = "ubuntu", stream: bool = True
) -> subprocess.CompletedProcess:
    """
    Execute a command on a remote host via SSH.

    Parameters
    ----------
    host : str
        The hostname or IP address.
    cmd : str
        The command to execute.
    user : str
        The SSH user to connect as.
    stream : bool
        If True, stream stdout/stderr to the terminal.

    Returns
    -------
    subprocess.CompletedProcess
        The result of the command.
    """
    ssh_cmd = [
        "ssh",
        "-o",
        "StrictHostKeyChecking=no",
        f"{user}@{host}",
        cmd,
    ]

    if stream:
        result = subprocess.run(ssh_cmd)
    else:
        result = subprocess.run(ssh_cmd, capture_output=True, text=True)

    return result


def rsync_to_remote(
    local_path: str, host: str, remote_path: str, user: str = "ubuntu"
) -> subprocess.CompletedProcess:
    """
    Sync a local directory to a remote host using rsync.

    Parameters
    ----------
    local_path : str
        The local directory path.
    host : str
        The hostname or IP address.
    remote_path : str
        The remote directory path.
    user : str
        The SSH user to connect as.

    Returns
    -------
    subprocess.CompletedProcess
        The result of the rsync command.
    """
    local_path = local_path.rstrip("/") + "/"

    rsync_cmd = [
        "rsync",
        "-avz",
        "--progress",
        "-e",
        "ssh -o StrictHostKeyChecking=no",
        local_path,
        f"{user}@{host}:{remote_path}",
    ]

    return subprocess.run(rsync_cmd)


def interactive_ssh(host: str, user: str = "ubuntu") -> None:
    """
    Open an interactive SSH session.

    Parameters
    ----------
    host : str
        The hostname or IP address.
    user : str
        The SSH user to connect as.
    """
    subprocess.run(
        ["ssh", "-o", "StrictHostKeyChecking=no", f"{user}@{host}"],
        stdin=sys.stdin,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
