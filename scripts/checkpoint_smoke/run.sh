#!/usr/bin/env bash
# End-to-end checkpoint smoke test.
#
# Runs the SDK scenarios (smoke.py) on HOST against a throwaway SeaweedFS S3
# server in Docker there, with real torch and the main branch's SDK for
# compatibility checks. Then runs the TUI checks (tui_checks.py) from this
# machine through an ssh tunnel. HOST needs Docker and uv (~/.local/bin/uv).
# Everything on HOST lives under ~/extty-smoke; its own ~/.extty is untouched.
#
# Usage: scripts/checkpoint_smoke/run.sh HOST
set -euo pipefail

HOST=${1:?usage: run.sh HOST}
REPO=$(git rev-parse --show-toplevel)
HERE="$REPO/scripts/checkpoint_smoke"
CONTAINER=extty-smoke-s3
BUCKET=extty-smoke

echo "=== syncing code to $HOST"
ssh "$HOST" 'mkdir -p ~/extty-smoke'
rsync -a --delete --exclude .venv --exclude __pycache__ --exclude '*.egg-info' \
    --exclude .pytest_cache "$REPO/sdk" "$REPO/spec" "$REPO/scripts" "$HOST:extty-smoke/new/"
git -C "$REPO" archive origin/main sdk |
    ssh "$HOST" 'rm -rf ~/extty-smoke/main && mkdir -p ~/extty-smoke/main && tar -x -C ~/extty-smoke/main'

echo "=== preparing Python and the S3 server on $HOST"
ssh "$HOST" bash -s <<REMOTE
set -euo pipefail
cd ~/extty-smoke
if [ ! -x venv/bin/python ]; then
    ~/.local/bin/uv venv -q --python 3.12 venv
    ~/.local/bin/uv pip install -q -p venv/bin/python psutil boto3 nvidia-ml-py numpy
    ~/.local/bin/uv pip install -q -p venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
fi
if [ -z "\$(docker ps -q --filter name=$CONTAINER)" ]; then
    docker rm -f $CONTAINER >/dev/null 2>&1 || true
    mkdir -p s3-data
    docker run -d --name $CONTAINER -p 127.0.0.1:9000:8333 -v "\$HOME/extty-smoke/s3-data:/data" \
        chrislusf/seaweedfs server -dir=/data -s3 -s3.port=8333 >/dev/null
fi
venv/bin/python - <<'PY'
import time, boto3
c = boto3.client("s3", endpoint_url="http://127.0.0.1:9000", aws_access_key_id="smoke",
                 aws_secret_access_key="smoke-secret", region_name="us-east-1")
for _ in range(60):
    try:
        c.head_bucket(Bucket="$BUCKET")
        break
    except c.exceptions.ClientError:
        c.create_bucket(Bucket="$BUCKET")
        break
    except Exception:
        time.sleep(1)
PY
REMOTE

echo "=== SDK scenarios on $HOST"
set +e
ssh "$HOST" "cd ~/extty-smoke && PYTHONPATH=\$HOME/extty-smoke/new/sdk venv/bin/python \
    new/scripts/checkpoint_smoke/smoke.py run --workdir \$HOME/extty-smoke/work \
    --endpoint http://127.0.0.1:9000 --bucket $BUCKET --container $CONTAINER \
    --main-sdk \$HOME/extty-smoke/main/sdk" 2>&1 |
    grep -E '^(PASS|FAIL|---|S3 prefix)|^      \(|passed'
SDK_STATUS=${PIPESTATUS[0]}
set -e

echo "=== TUI checks from $(hostname)"
cargo build --quiet --manifest-path "$REPO/tui/Cargo.toml"
set +e
python3 "$HERE/tui_checks.py" --host "$HOST" \
    --remote-workdir "$(ssh "$HOST" 'echo $HOME')/extty-smoke/work" \
    --extty "$REPO/tui/target/debug/extty" --tui-dir "$REPO/tui"
TUI_STATUS=$?
set -e

echo "=== SDK exit $SDK_STATUS, TUI exit $TUI_STATUS"
exit $(( SDK_STATUS || TUI_STATUS ))
