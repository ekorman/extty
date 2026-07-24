import math
import random
import time

from PIL import Image, ImageDraw

import extty


def make_frame(step: int, width: int = 480, height: int = 320) -> Image.Image:
    """Render a gradient frame with a few fake detection boxes drawn on it."""
    img = Image.new("RGB", (width, height))
    for y in range(height):
        for x in range(0, width, 4):
            r = int(255 * x / width)
            g = int(255 * y / height)
            b = (step * 25) % 255
            for dx in range(4):
                img.putpixel((min(x + dx, width - 1), y), (r, g, b))

    draw = ImageDraw.Draw(img)
    rng = random.Random(step)
    for i in range(3):
        x0 = rng.randint(0, width - 120)
        y0 = rng.randint(0, height - 90)
        x1, y1 = x0 + rng.randint(60, 120), y0 + rng.randint(45, 90)
        draw.rectangle((x0, y0, x1, y1), outline="red", width=3)
        draw.text((x0 + 4, y0 + 4), f"obj{i} {rng.random():.2f}", fill="white")
    draw.text((10, 10), f"step {step}", fill="yellow")
    return img


if __name__ == "__main__":
    run = extty.init("image-demo", config={"model": "fake-detector"})
    try:
        for step in range(10):
            loss = 2.0 / (step + 1) + 0.05 * math.sin(step)
            extty.log(
                {
                    "train/loss": loss,
                    "val/detections": extty.Image(
                        make_frame(step), caption=f"detections at step {step}"
                    ),
                },
                step=step,
            )
            time.sleep(0.2)
    finally:
        extty.finish()
