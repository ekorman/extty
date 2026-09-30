# extty

Terminal-first ML experiment tracking: a Python SDK for logging runs and a
Rust TUI for browsing them.

```bash
pip install extty            # extty[image] for Image logging
cargo install --path tui     # optional: the `extty` TUI/CLI
```

The TUI and S3 are both optional. Runs are stored locally, and the Python SDK
alone can log and read them. Configure S3 to sync runs to a bucket.

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
From Python: `extty.push("my-project/", dry_run=True)`.

## Where data is stored

Runs, checkpoints, the artifact cache and the S3 config all live under
`~/.extty`. Set `EXTTY_HOME` to move them, e.g. to scratch space on a cluster
with a small home quota. The SDK and TUI both honor it.

```bash
export EXTTY_HOME=/scratch/$USER/extty
```
