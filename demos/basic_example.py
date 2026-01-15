import extty
import time
import math

if __name__ == "__main__":
    run = extty.init(
        "demo-project",
        server=True,
        config={"lr": 0.001, "batch_size": 32, "model": "bert-base"},
    )

    print(f"server info: {run.server_info}")
    try:
        for i in range(1000):
            loss = 2.0 / (i + 1) + 0.1 * math.sin(i / 10)
            acc = min(0.99, 0.3 + i * 0.0007 + 0.02 * math.cos(i / 8))
            val_loss = loss * 1.1 + 0.05 * math.sin(i / 15)

            extty.log(
                {
                    "train/loss": loss,
                    "train/acc": acc,
                    "val/loss": val_loss,
                    "val/example": extty.Example(
                        prompt=f"prompt{i}", responses=[f"response{i}"]
                    ),
                },
                step=i,
            )

            time.sleep(0.3)
    finally:
        extty.finish()
