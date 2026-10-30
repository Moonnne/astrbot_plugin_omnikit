"""Validated voice parameters and isolated WORLD resynthesis."""

import json
import math
import sys
import wave

# Key, display label, neutral value, minimum and maximum.
PARAMETERS = {
    "pitch": ("音高偏移（半音）", 0, -12, 12),
    "target_f0": ("目标基频（Hz，0保持）", 0, 0, 600),
    "variation": ("语调起伏（倍）", 1, 0, 2),
    "smooth_ms": ("语调平滑（毫秒）", 0, 0, 80),
    "formant": ("共振峰（倍）", 1, 0.7, 1.4),
    "breath": ("气声（倍）", 1, 0, 2),
    "tilt": ("频谱倾斜（dB每倍频程）", 0, -6, 6),
    "metal": ("金属感（0至1）", 0, 0, 1),
    "metal_hz": ("金属调制频率（Hz）", 45, 10, 200),
    "wet": ("效果混合（0至1）", 1, 0, 1),
    "speed": ("语速（倍）", 1, 0.5, 2),
    "bass": ("低频增益（dB）", 0, -12, 12),
    "treble": ("高频增益（dB）", 0, -12, 12),
    "gain": ("输出增益（dB）", 0, -18, 12),
    "f0_floor": ("基频检测下限（Hz）", 60, 50, 200),
    "f0_ceil": ("基频检测上限（Hz）", 600, 200, 1000),
}
NEUTRAL = {**{key: spec[1] for key, spec in PARAMETERS.items()}, "method": "harvest"}
BUILTINS = {
    "中性": NEUTRAL,
    "GLaDOS": {
        **NEUTRAL,
        "pitch": -1,
        "variation": 0.65,
        "smooth_ms": 15,
        "formant": 1.08,
        "breath": 0.8,
        "metal": 0.12,
        "treble": 2,
        "speed": 0.98,
    },
}


def number(value, name, low, high):
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ValueError(f"{name} 必须为 {low}–{high} 的数字")
    try:
        result = float(value)
    except ValueError:
        raise ValueError(f"{name} 必须为 {low}–{high} 的数字") from None
    if not math.isfinite(result) or not low <= result <= high:
        raise ValueError(f"{name} 必须为 {low}–{high} 的数字")
    return result


def validate_params(values):
    if not isinstance(values, dict) or set(values) != set(PARAMETERS) | {"method"}:
        raise ValueError("音频预设参数不完整或包含未知参数")
    result = {
        key: number(values[key], label, low, high)
        for key, (label, _, low, high) in PARAMETERS.items()
    }
    result["method"] = values["method"]
    if result["method"] not in ("harvest", "dio"):
        raise ValueError("基频算法须为 harvest 或 dio")
    if result["f0_floor"] >= result["f0_ceil"]:
        raise ValueError("基频检测下限必须小于上限")
    if 0 < result["target_f0"] < 50:
        raise ValueError("目标基频须为 0 或 50–600 Hz")
    return result


def render(source, output, params):
    """Resynthesize mono PCM using WORLD without importing AstrBot.

    Args:
        source: Path to a bounded 24 kHz mono PCM16 WAV file.
        output: Path to the resulting PCM16 WAV file.
        params: Validated effect parameter mapping.
    """
    import numpy as np
    import pyworld as world

    p = validate_params(params)
    with wave.open(str(source), "rb") as reader:
        if (
            reader.getnchannels() != 1
            or reader.getsampwidth() != 2
            or reader.getframerate() != 24000
            or not 2400 <= reader.getnframes() <= 24000 * 180
        ):
            raise ValueError("Unsupported WORLD input")
        x = np.frombuffer(reader.readframes(reader.getnframes()), "<i2")
        x = x.astype(np.float64) / 32768
    rate, step = 24000, 5.0
    y = x.copy()
    if np.max(np.abs(x)) > 1e-5 and p["wet"] > 0:
        estimator = world.harvest if p["method"] == "harvest" else world.dio
        f0, times = estimator(
            x, rate, f0_floor=p["f0_floor"], f0_ceil=p["f0_ceil"], frame_period=step
        )
        f0 = world.stonemask(x, f0, times, rate)
        spectrum = world.cheaptrick(x, f0, times, rate, f0_floor=p["f0_floor"])
        noise = world.d4c(x, f0, times, rate, fft_size=(spectrum.shape[1] - 1) * 2)
        voiced = f0 > 0
        if np.any(voiced):
            logs = np.log2(f0[voiced])
            center = np.median(logs)
            target = math.log2(p["target_f0"]) if p["target_f0"] else center
            f0[voiced] = 2 ** (
                target + (logs - center) * p["variation"] + p["pitch"] / 12
            )
            width = max(1, round(p["smooth_ms"] / step))
            # Smooth within voiced runs so consonants never acquire a pitch.
            edges = np.flatnonzero(np.diff(np.r_[False, voiced, False]))
            for start, end in zip(edges[::2], edges[1::2]):
                size = min(width, end - start)
                values = np.log(f0[start:end])
                padded = np.pad(values, (size // 2, (size - 1) // 2), mode="edge")
                f0[start:end] = np.exp(
                    np.convolve(padded, np.ones(size) / size, "valid")
                )
            f0[voiced] = np.clip(f0[voiced], 40, 1200)
        bins = np.arange(spectrum.shape[1], dtype=np.float64)
        if p["formant"] != 1:
            for row in spectrum:
                row[:] = np.interp(bins / p["formant"], bins, row)
        frequencies = bins * rate / ((len(bins) - 1) * 2)
        tilt_db = np.clip(
            p["tilt"] * np.log2(np.maximum(frequencies, 80) / 1000), -24, 24
        )
        spectrum *= 10 ** (tilt_db / 10)
        noise[voiced] = np.clip(noise[voiced] * p["breath"], 0.001, 1)
        y = world.synthesize(f0, spectrum, noise, rate, frame_period=step)
        y = np.pad(y[: len(x)], (0, max(0, len(x) - len(y))))
        if p["metal"]:
            carrier = np.sin(2 * np.pi * p["metal_hz"] * np.arange(len(y)) / rate)
            y *= 1 - p["metal"] + p["metal"] * carrier
        y = x * (1 - p["wet"]) + y * p["wet"]
    if not np.isfinite(y).all():
        raise ValueError("Non-finite WORLD output")
    peak = np.max(np.abs(y))
    if peak > 0.98:
        y *= 0.98 / peak
    with wave.open(str(output), "wb") as writer:
        writer.setparams((1, 2, rate, 0, "NONE", "not compressed"))
        writer.writeframes(np.rint(np.clip(y, -1, 1) * 32767).astype("<i2").tobytes())


if __name__ == "__main__":
    render(sys.argv[1], sys.argv[2], json.loads(sys.argv[3]))
