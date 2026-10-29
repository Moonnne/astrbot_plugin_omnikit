"""Independent JavaScript vectors and image quality contracts."""

import json
from dataclasses import replace
from io import BytesIO
from pathlib import Path
from threading import Event

import pytest
from PIL import Image

from core.config import Settings
from core.image import curve, transform
from core.text import decode, encode, parse_text, validate_alphabet

REFERENCE = json.loads(
    (Path(__file__).parent / "fixtures/reference.json").read_text(encoding="utf-8")
)


@pytest.mark.parametrize("item", REFERENCE["texts"])
def test_beast_javascript_vectors(item):
    assert encode(item["plain"], "嗷呜啊~") == item["cipher"]
    assert decode(item["cipher"]) == item["plain"]


@pytest.mark.parametrize("alphabet", ["嗡呜隆嗷", "我是肥鱼", "🐋🐟🍚🐠", "a=~!"])
def test_custom_alphabet(alphabet):
    raw = "🐋 肥鱼\n吃\t白饭𠮷"
    assert decode(encode(raw, alphabet)) == raw
    assert decode(encode(raw, alphabet), alphabet) == raw


@pytest.mark.parametrize(
    "alphabet",
    ["三字符", "五个字符啊", "aaaa", "啊啊呜嗷", "abc ", "abc\u200b", "abc\ud800"],
)
def test_invalid_alphabet(alphabet):
    with pytest.raises(ValueError):
        validate_alphabet(alphabet)


def test_text_parser_and_errors():
    presets = {"兽音": "嗷呜啊~"}
    assert parse_text("你好", presets) == (None, "你好")
    assert parse_text("-- hello  world\nhi", presets) == (None, "hello  world\nhi")
    assert parse_text("我是肥鱼 hi  there", presets) == ("我是肥鱼", "hi  there")
    assert parse_text("兽音 hi", presets) == ("嗷呜啊~", "hi")
    for raw in ("", "--", "佛曰 hi", "abc hi", "abcde hi"):
        with pytest.raises(ValueError):
            parse_text(raw, presets)
    raw = encode("鲸鱼")
    for bad in (raw[:-1], raw + "啊", raw[:4] + "外" + raw[5:], "嗷嗷嗷嗷"):
        with pytest.raises(ValueError):
            decode(bad)
    with pytest.raises(ValueError):
        decode(raw, "我是肥鱼")


def png(width=7, height=5, mode="RGBA"):
    image = Image.new("RGBA", (width, height))
    image.putdata(
        [
            (i % 256, i * 13 % 256, i * 23 % 256, i * 47 % 256)
            for i in range(width * height)
        ]
    )
    output = BytesIO()
    image.convert(mode).save(output, "PNG")
    return output.getvalue()


def pixels(raw):
    with Image.open(BytesIO(raw)) as image:
        return image.size, image.convert("RGBA").tobytes()


def test_all_reference_curves_and_forward_mapping():
    for item in REFERENCE["curves"]:
        w, h, path = item["w"], item["h"], item["path"]
        assert list(curve(w, h)) == path
        assert sorted(path) == list(range(w * h))
    item = next(v for v in REFERENCE["curves"] if v["w"] == 7 and v["h"] == 5)
    original = pixels(png())[1]
    output = bytearray(len(original))
    path = item["path"]
    for i, src in enumerate(path):
        dst = path[(i + item["offset"]) % len(path)]
        output[dst * 4 : dst * 4 + 4] = original[src * 4 : src * 4 + 4]
    assert pixels(transform(png(), 1, False, Settings(), Event()))[1] == bytes(output)


@pytest.mark.parametrize("size", [(1, 1), (1, 19), (19, 1), (7, 5), (6, 9), (11, 11)])
@pytest.mark.parametrize("rounds", [1, 3, 10])
def test_lossless_roundtrip(size, rounds):
    original = png(*size)
    result = transform(original, rounds, False, Settings(), Event())
    assert result.startswith(b"\x89PNG\r\n\x1a\n")
    assert pixels(transform(result, rounds, True, Settings(), Event())) == pixels(
        original
    )


def test_repeated_mapping_and_limits():
    raw = png()
    repeated = raw
    for _ in range(3):
        repeated = transform(repeated, 1, False, Settings(), Event())
    assert pixels(repeated) == pixels(transform(raw, 3, False, Settings(), Event()))
    for bad, cfg in (
        (b"invalid", Settings()),
        (raw, replace(Settings(), max_pixels=10)),
        (raw, replace(Settings(), max_bytes=1)),
    ):
        with pytest.raises(ValueError):
            transform(bad, 1, False, cfg, Event())
    stop = Event()
    stop.set()
    with pytest.raises(ValueError, match="取消"):
        transform(raw, 1, False, Settings(), stop)


def test_animation_rejected():
    out = BytesIO()
    Image.new("RGB", (2, 2), "red").save(
        out, "GIF", save_all=True, append_images=[Image.new("RGB", (2, 2), "blue")]
    )
    with pytest.raises(ValueError, match="静态"):
        transform(out.getvalue(), 1, False, Settings(), Event())


def test_settings_and_schema():
    schema = json.loads(
        (Path(__file__).parents[1] / "_conf_schema.json").read_text(encoding="utf-8")
    )
    cfg = Settings.parse({k: v["default"] for k, v in schema.items()})
    assert cfg == Settings()
    assert cfg.wait_seconds == 60
    for bad in (
        {"wait_seconds": True},
        {"presets": ["兽音=aaaa"]},
        {"alphabet": "五个字太多"},
        {"presets": ["兽音=嗷呜啊~", "兽音=我是肥鱼"]},
    ):
        with pytest.raises(ValueError):
            Settings.parse(bad)
