"""Gilbert-path pixel rotation, independently expressed using standard arrays."""

import warnings
from array import array
from io import BytesIO
from math import floor, sqrt
from threading import Event

from PIL import Image, ImageOps, UnidentifiedImageError

from .config import Settings
from .errors import UserError


def curve(width: int, height: int):
    """Yield row-major indices in the website's generalized Hilbert order."""

    def walk(x, y, ax, ay, bx, by):
        w, h = abs(ax + ay), abs(bx + by)
        dx, dy = (ax > 0) - (ax < 0), (ay > 0) - (ay < 0)
        ex, ey = (bx > 0) - (bx < 0), (by > 0) - (by < 0)
        if h == 1:
            for _ in range(w):
                yield x + y * width
                x, y = x + dx, y + dy
            return
        if w == 1:
            for _ in range(h):
                yield x + y * width
                x, y = x + ex, y + ey
            return
        a, b, c, d = ax // 2, ay // 2, bx // 2, by // 2
        if 2 * w > 3 * h:
            if abs(a + b) % 2 and w > 2:
                a, b = a + dx, b + dy
            yield from walk(x, y, a, b, bx, by)
            yield from walk(x + a, y + b, ax - a, ay - b, bx, by)
        else:
            if abs(c + d) % 2 and h > 2:
                c, d = c + ex, d + ey
            yield from walk(x, y, c, d, a, b)
            yield from walk(x + c, y + d, ax, ay, bx - c, by - d)
            yield from walk(
                x + ax - dx + c - ex, y + ay - dy + d - ey, -c, -d, -(ax - a), -(ay - b)
            )

    if width < 1 or height < 1:
        raise UserError("图片尺寸必须为正数。")
    yield from (
        walk(0, 0, width, 0, 0, height)
        if width >= height
        else walk(0, 0, 0, height, width, 0)
    )


def transform(
    data: bytes, rounds: int, reverse: bool, settings: Settings, stop: Event
) -> bytes:
    """Return lossless PNG, preserving the decoded and oriented RGBA pixels."""

    def check():
        if stop.is_set():
            raise UserError("图片处理已取消或超时。")

    if not 1 <= rounds <= settings.max_rounds:
        raise UserError("混淆次数超出范围。")
    if len(data) > settings.max_bytes:
        raise UserError("图片文件过大。")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(data)) as source:
                w, h = source.size
                if w * h > settings.max_pixels or max(w, h) > settings.max_side:
                    raise UserError(
                        "图片尺寸超限，请提供原尺寸较小的图片；本鲸不会自动缩图。"
                    )
                if getattr(source, "n_frames", 1) != 1:
                    raise UserError("首版只处理静态图片，请先转换动画或多页图片。")
                check()
                rgba = ImageOps.exif_transpose(source).convert("RGBA")
        w, h = rgba.size
        pixels = array("I")
        pixels.frombytes(rgba.tobytes())
        rgba.close()
        if pixels.itemsize != 4:
            raise UserError("当前系统不支持四字节像素数组。")
        n = w * h
        path = array("I")
        for i, index in enumerate(curve(w, h)):
            if i % 4096 == 0:
                check()
            path.append(index)
        shift = floor((sqrt(5) - 1) / 2 * n + 0.5) * rounds % n
        if reverse:
            shift = -shift
        output = array("I", [0]) * n
        for i, index in enumerate(path):
            if i % 4096 == 0:
                check()
            output[path[(i + shift) % n]] = pixels[index]
        check()
        with Image.frombytes("RGBA", (w, h), output.tobytes()) as result:
            stream = BytesIO()
            result.save(stream, format="PNG")
        check()
        value = stream.getvalue()
        if len(value) > settings.max_bytes:
            raise UserError("无损结果文件超限，请使用尺寸较小的原图。")
        return value
    except (
        UnidentifiedImageError,
        OSError,
        Image.DecompressionBombWarning,
        Image.DecompressionBombError,
    ) as exc:
        raise UserError("无法读取这张图片，文件可能损坏或尺寸过大。") from exc
