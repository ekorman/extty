"""Dataclass-like wrapper for logging images."""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image as PILImage


class Image:
    """
    An image to log, wrapping a ``PIL.Image.Image``.

    The image is encoded to PNG eagerly at construction time, on the caller's
    thread, so the object handed to the storage sinks carries only plain bytes
    and metadata — no PIL objects cross the async boundary and pillow is never
    required to read images back.

    Requires pillow (``pip install extty[image]``). Note that the encoded PNG
    bytes are held in memory until flushed to storage, so logging very large
    images at high frequency will increase memory usage accordingly.

    Parameters
    ----------
    image : PIL.Image.Image
        The image to log. Draw any annotations (e.g. bounding boxes via
        ``PIL.ImageDraw``) before constructing the ``Image``.
    caption : str, optional
        A short caption stored alongside the image.
    """

    __slots__ = ("png_bytes", "width", "height", "mode", "caption")

    def __init__(self, image: PILImage.Image, caption: str | None = None) -> None:
        try:
            from PIL import Image as PILImageModule
        except ImportError as exc:
            raise TypeError(
                "extty.Image requires a PIL.Image.Image, but pillow is not "
                "installed. Install it with `pip install extty[image]` or "
                "`pip install pillow`."
            ) from exc
        if not isinstance(image, PILImageModule.Image):
            raise TypeError(
                f"extty.Image expects a PIL.Image.Image, got {type(image).__name__}"
            )

        buf = io.BytesIO()
        try:
            image.save(buf, format="PNG")
        except (OSError, ValueError):
            buf = io.BytesIO()
            image.convert("RGB").save(buf, format="PNG")

        self.png_bytes: bytes = buf.getvalue()
        self.width: int = image.width
        self.height: int = image.height
        self.mode: str = image.mode
        self.caption: str | None = caption

    def __repr__(self) -> str:
        return (
            f"Image(width={self.width}, height={self.height}, mode={self.mode!r}, "
            f"caption={self.caption!r}, png_bytes=<{len(self.png_bytes)} bytes>)"
        )
