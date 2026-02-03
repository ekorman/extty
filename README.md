S3 Sync Usage

Python SDK

Setup: Install with S3 support
pip install extty[s3]

Automatic S3 upload during training: Set environment variables
export EXTTY_S3_BUCKET=my-bucket
export EXTTY_S3_PREFIX=extty          # optional, default: "extty"
export EXTTY_S3_REGION=us-west-2      # optional
export EXTTY_S3_ENDPOINT_URL=...      # optional, for MinIO etc.

Then use extty normally - metrics automatically sync to S3:
import extty

run = extty.init("my-project", name="run-1")
run.log({"loss": 0.5}, step=0)
run.finish()

Manual push from Python:
import extty

extty.push("my-project/run-1")        # single run
extty.push("my-project/")             # all runs in project
extty.push()                          # all runs
extty.push("my-project/", dry_run=True)   # preview
extty.push("my-project/", force=True)     # overwrite, no merge

---
Rust TUI/CLI

Setup: Create ~/.extty/s3/config.toml
bucket = "my-bucket"
prefix = "extty"
region = "us-west-2"
access_key_id = "AKIA..."        # optional, uses AWS cred chain
secret_access_key = "..."
endpoint_url = "..."             # optional, for MinIO etc.

Commands:
extty pull project/run-name      # download single run from S3
extty pull project/              # download all runs in project
extty pull                       # download all runs

extty push project/run-name      # upload single run to S3
extty push project/              # upload all runs in project
extty push                       # upload all runs

extty sync project/run-name      # bidirectional sync (pull then push)

# Options:
extty pull -n                    # --dry-run: preview without doing
extty push -f                    # --force: overwrite without merging

Launch TUI (unchanged):
extty                            # opens TUI dashboard
