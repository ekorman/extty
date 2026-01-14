import extty
import time
import math
import modal

app = modal.App("example")
image = modal.Image.debian_slim().uv_sync().add_local_python_source("extty")


# @app.function()
# def foo():
#     print(locals())
#     print("a", modal.current_function_call_id())
#     # with modal.forward(8000) as tunnel:
#     #     import pdb

#     #     pdb.set_trace()
#     return "bar"


@app.function(image=image)
def main():
    extty.init(
        "demo-modal-project",
        server=True,
        config={"lr": 0.001, "batch_size": 32, "model": "bert-base"},
        create_modal_tunnel=True,
    )

    for i in range(1000):
        loss = 2.0 / (i + 1) + 0.1 * math.sin(i / 10)
        acc = min(0.99, 0.3 + i * 0.0007 + 0.02 * math.cos(i / 8))
        val_loss = loss * 1.1 + 0.05 * math.sin(i / 15)

        extty.log(
            {
                "train/loss": loss,
                "train/acc": acc,
                "val/loss": val_loss,
                "val/example": {"prompt": f"prompt{i}", "response": f"response{i}"},
            },
            step=i,
        )

        time.sleep(0.3)

    extty.finish()
