# extty

Terminal-first ML experiment tracking: a Python SDK for logging runs and a
Rust TUI for browsing them.

The TUI and S3 are both optional. Runs are stored locally, and the Python SDK
alone can log and read them. Configure S3 to sync runs to a bucket.

## Install

### Python SDK

Requires Python 3.10+.

```bash
pip install extty            # or: uv add extty
pip install "extty[image]"   # adds Image logging
```

### TUI

Prebuilt binaries for macOS, Linux and Windows are attached to each
[GitHub release](https://github.com/ekorman/extty/releases). On macOS or
Linux, set `target` to your platform and extract `extty` onto your `PATH`:

```bash
# aarch64-apple-darwin, x86_64-apple-darwin,
# x86_64-unknown-linux-gnu or aarch64-unknown-linux-gnu
target=aarch64-apple-darwin
mkdir -p ~/.local/bin
curl -LsSf "https://github.com/ekorman/extty/releases/latest/download/extty-$target.tar.xz" \
  | tar xJ --strip-components=1 -C ~/.local/bin "extty-$target/extty"
```

On Windows, download `extty-x86_64-pc-windows-msvc.zip` from the release and
put `extty.exe` on your `PATH`.

To build from source instead:

```bash
cargo install --locked --git https://github.com/ekorman/extty extty
```

## Logging

```python
import extty

extty.init("my-project", name="run-1", config={"lr": 1e-3})
extty.log({"train/loss": 0.5}, step=0)
extty.finish()
```

The value's type decides how it is stored and rendered. Use `/` in names to
group streams (`train/loss`, `val/loss`).

| Type | For | TUI |
| --- | --- | --- |
| `float` / `int` | scalar metrics | line chart |
| `Example` / `BatchExample` | prompt/response pairs, with optional rewards (float or dict of components) and groundtruth | text browser |
| `ConfusionMatrix` | N×N counts, rows = true class | matrix |
| `Chart` | `(x, y)` curves such as ROC or PR | plot; `v` toggles a table |
| `Image` | a `PIL.Image`, stored as PNG; needs `extty[image]` | inline on kitty/iTerm2/sixel; `o` opens externally |

```python
extty.log(
    {
        "train/loss": loss,
        "val/samples": extty.Example(
            prompt="Capital of France?",
            responses=["Paris", "Lyon"],
            rewards=[1.0, 0.0],
            groundtruth="Paris",
        ),
        "eval/cm": extty.ConfusionMatrix.from_array(cm, labels=["cat", "dog"]),
        "eval/roc": extty.Chart.from_arrays(fpr, tpr, ("fpr", "tpr")),
        "val/detections": extty.Image(frame, caption="epoch 3"),
    },
    step=step,
)
```

`BatchExample` takes parallel lists, one entry per prompt. `from_array` and
`from_arrays` accept numpy or torch arrays.

In the TUI run view, focus a card with Enter and scrub through steps with
`↑↓` (Shift jumps 10).

## Reading runs

```python
run = extty.get_run("my-project", "run-1")
run.metric_names
run.metric("train/loss")          # [MetricPoint(step, timestamp, value), ...]
run.examples("val/samples")
run.confusion_matrix("eval/cm")
run.chart("eval/roc")
run.image_bytes(run.images("val/detections")[-1])
```

With S3 configured, `get_run` reads a run straight from S3 if it isn't local.

## Checkpoints

```python
extty.save_checkpoint(step, state_dict=model.state_dict(), optimizer_state_dict=opt.state_dict())
extty.save_checkpoint(step, path="ckpt.pt")   # or a file written by torch.save

ckpt = extty.load_checkpoint(step)                              # active run
ckpt = extty.load_checkpoint_from("my-project", "run-1", step)  # any run
model.load_state_dict(ckpt["model_state_dict"])
```

Checkpoints are saved in the run directory, so they work without S3. With S3
configured they are uploaded instead, and kept locally only with
`keep_local=True` or if the upload fails. Loads use a complete local copy when
there is one and download from S3 otherwise. `extty push` does not upload
checkpoints.

`extty.delete_local_checkpoint(project, run, step)` and `extty prune local`
free disk by deleting local copies that S3 also has. Neither deletes a
checkpoint's only copy unless you pass `force=True` to
`delete_local_checkpoint`.

`save_checkpoint` flushes the rows logged so far before saving, and returns
`True` only when the save reached S3, `False` when it was kept only locally
(no S3, or a failed upload).

### Resuming a run

To continue a run after its job dies, start it again under the same name with
`resume=True`, on the same machine or, with S3 configured, on a fresh one:

```python
run = extty.init("my-project", name="job-1234", config=config, resume=True)
start = run.resume_step or 0       # None for a new run or no checkpoint yet
if run.resume_step is not None:
    ckpt = extty.load_checkpoint(run.resume_step)
    model.load_state_dict(ckpt["model_state_dict"])
    opt.load_state_dict(ckpt["optimizer_state_dict"])
for step in range(start + 1, n_steps + 1):
    ...
```

The run is read from S3 when configured and from the run directory otherwise.
If it exists:

- Its stored config must match `config`, keys starting with `_` (git hash,
  GPU, instance) aside. A difference raises `extty.RunConfigMismatch`, listing
  the dotted paths that differ, before anything is written.
- Its logged metrics, examples, confusion matrices, charts and image index are
  cut back to its latest checkpoint, `run.resume_step`, in S3 and locally, and
  new rows append to them. With no checkpoint, they are emptied. `system.csv`,
  image files and checkpoints are kept.
- `config` and `started_at` stay as first stored, and `meta.json` records each
  resume under `resumes`.

`run.resumed` says whether an existing run was continued; with no such run,
`resume=True` starts a new one. Only one process may write a run name at a
time, and resuming is not supported in distributed training. Without
`resume=True`, a run with the same name is overwritten.

With S3 configured, `resume=True` never falls back to a new run because of an
error: init raises if reading the run from S3 fails, if its `meta.json` cannot
be written to S3, or if S3 holds files of the run but no `meta.json`.

## S3 sync (optional)

Configure S3 in `~/.extty/s3/config.toml`, which both the SDK and TUI read:

```toml
bucket = "my-bucket"
prefix = "extty"          # optional
region = "us-west-2"      # optional
access_key_id = "..."     # optional, defaults to the AWS credential chain
secret_access_key = "..."
endpoint_url = "..."      # optional, for MinIO etc.
```

The SDK also accepts `EXTTY_S3_BUCKET`, `EXTTY_S3_PREFIX`, `EXTTY_S3_REGION`,
`EXTTY_S3_ACCESS_KEY_ID`, `EXTTY_S3_SECRET_ACCESS_KEY` and
`EXTTY_S3_ENDPOINT_URL`, which take precedence over the file. Once configured,
runs sync to S3 as they log. Without it, the SDK makes no network requests.

With the TUI installed:

```bash
extty                     # open the TUI
extty pull [project/[run]]
extty push [project/[run]]
extty sync project/run    # pull, then push
extty prune local         # delete local checkpoints/artifacts that S3 has
```

`-n`/`--dry-run` previews, and `-f`/`--force` overwrites instead of merging.
A run resumed since it was pulled is not merged row by row, since the resume
cut its history back: a pull replaces the local copy's history with S3's (even
for a run marked completed), and a push from the older copy is refused until it
is pulled again.
From Python: `extty.push("my-project/", dry_run=True)`.

## Where data is stored

Runs, checkpoints, the artifact cache and the S3 config all live under
`~/.extty`. Set `EXTTY_HOME` to move them, e.g. to scratch space on a cluster
with a small home quota. The SDK and TUI both honor it.

```bash
export EXTTY_HOME=/scratch/$USER/extty
```
