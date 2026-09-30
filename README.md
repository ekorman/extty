# extty: terminal first MLOps

# Logging

Everything is logged through `extty.log`, which takes a dict of names to
values plus a `step`. The value's type determines how it is stored and how
the TUI renders it:

```python
import extty

run = extty.init("my-project", name="run-1", config={"lr": 1e-3})
extty.log({"train/loss": 0.5}, step=0)
extty.finish()
```

Names may contain `/` to group related streams (e.g. `train/loss`,
`val/loss`). Each type below gets its own card in the TUI run view; focus a
card with Enter and scrub through steps with `↑↓` (Shift jumps 10).

## Scalar metrics

Any `float` or `int` is logged as a time-series metric and plotted as a line
chart.

```python
extty.log({"train/loss": loss, "val/acc": 0.93}, step=step)
```

## Text examples (`Example` / `BatchExample`)

Prompt/response pairs, e.g. for LLM training or evals. Each response can
carry an optional reward (a float, or a dict of named reward components),
and each prompt an optional groundtruth answer. The TUI shows a browsable
prompt/response/groundtruth panel.

```python
extty.log(
    {
        "val/samples": extty.Example(
            prompt="Capital of France?",
            responses=["Paris", "Lyon"],
            rewards=[1.0, 0.0],
            groundtruth="Paris",
        )
    },
    step=step,
)

# Batched variant: one entry per prompt
extty.log(
    {
        "val/samples": extty.BatchExample(
            prompts=["1+1?", "2+2?"],
            responses=[["2"], ["4", "5"]],
            rewards=[[1.0], [{"correct": 1.0, "format": 0.5}, {"correct": 0.0, "format": 0.5}]],
            groundtruth=["2", "4"],
        )
    },
    step=step,
)
```

## Confusion matrices (`ConfusionMatrix`)

An N×N integer matrix with class labels (rows = true class, columns =
predicted). `from_array` accepts numpy/torch arrays.

```python
extty.log(
    {"eval/cm": extty.ConfusionMatrix(matrix=[[5, 1], [0, 6]], labels=["cat", "dog"])},
    step=step,
)

extty.log({"eval/cm": extty.ConfusionMatrix.from_array(cm_array, labels)}, step=step)
```

## 2D charts (`Chart`)

An ordered series of `(x, y)` points with axis names — for per-step curves
like ROC or precision/recall that aren't a single scalar over time.
`from_arrays` accepts numpy/torch arrays. The TUI renders a plot with a
table toggle (`v`).

```python
extty.log(
    {"eval/roc": extty.Chart(points=[(0.0, 0.0), (0.5, 0.7), (1.0, 1.0)], axis_names=("fpr", "tpr"))},
    step=step,
)

extty.log({"eval/roc": extty.Chart.from_arrays(fpr, tpr, ("fpr", "tpr"))}, step=step)
```

## Images (`Image`)

Wraps a `PIL.Image` and stores it as a PNG — e.g. detection frames with
bounding boxes drawn on (draw annotations yourself with `PIL.ImageDraw`
before logging). Requires pillow (`pip install extty[image]`). The TUI
renders images inline in terminals with graphics support (kitty, iTerm2,
sixel) and falls back to a metadata panel elsewhere; `o` opens the current
image in the system viewer. Logging the same name/step again overwrites
that step's image.

```python
from PIL import Image, ImageDraw

frame = Image.fromarray(pixels)
draw = ImageDraw.Draw(frame)
draw.rectangle((x0, y0, x1, y1), outline="red", width=3)

extty.log({"val/detections": extty.Image(frame, caption="epoch 3 detections")}, step=step)
```

## Reading data back

```python
run = extty.get_run("my-project", "run-1")
run.metric_names            # ["train/loss", ...]
run.metric("train/loss")    # [MetricPoint(step, timestamp, value), ...]
run.examples("val/samples")
run.confusion_matrix("eval/cm")
run.chart("eval/roc")
run.images("val/detections")            # [ImageRecord(step, file, width, height, caption), ...]
run.image_bytes(run.images("val/detections")[-1])  # PNG bytes
```

## Checkpoints

Checkpoints are saved into the run directory, at
`<extty home>/runs/<project>/<run>/checkpoints/<step>/`, so they work with no
S3 configured.

```python
extty.save_checkpoint(step, state_dict=model.state_dict(), optimizer_state_dict=opt.state_dict())
extty.save_checkpoint(step, path="ckpt.pt")   # or a file written by torch.save

ckpt = extty.load_checkpoint(step)                              # active run
ckpt = extty.load_checkpoint_from("my-project", "run-1", step)  # any run
model.load_state_dict(ckpt["model_state_dict"])
```

With S3 configured (see below), each checkpoint is uploaded instead, and kept
locally only if you pass `keep_local=True` or the upload fails (an error is
logged with the local path), so a flaky bucket never costs you a checkpoint.
Loads use a complete local copy when there is one and download from S3
otherwise; a checkpoint downloaded once reloads without contacting S3.
Checkpoints reach S3 only through `save_checkpoint`: `extty push` does not
upload them.

`extty.delete_local_checkpoint(project, run, step)` frees disk by deleting a
local copy that is also in S3. It refuses to delete a checkpoint's only copy
unless you pass `force=True`.

## Where data is stored

Everything extty keeps locally (runs, checkpoints, the artifact cache, and the
S3 config file) lives under `~/.extty`. Set `EXTTY_HOME` to move it, e.g. to
scratch space on a cluster with a small home quota:

```bash
export EXTTY_HOME=/scratch/$USER/extty
```

The Python SDK and the `extty` TUI both honor it, and `extty run` installs the
S3 config under the remote host's `EXTTY_HOME` (readable only by you). The
Python SDK makes no network requests unless S3 is configured.

---

# S3 Sync Usage

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
