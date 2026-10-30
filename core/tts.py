"""Preset storage and bounded subprocess audio processing."""

import asyncio
import hashlib
import json
import os
import subprocess
import sys
import unicodedata
import uuid
import wave
from pathlib import Path

from .audio import BUILTINS, validate_params


class Presets:
    def __init__(self, root: Path):
        self.root = root / "tts" / "presets"

    def path(self, name: str) -> Path:
        if (
            not isinstance(name, str)
            or not 1 <= len(name) <= 40
            or name != name.strip()
            or any(
                c in '/\\<>:"|?*' or unicodedata.category(c).startswith("C")
                for c in name
            )
        ):
            raise ValueError("预设名称须为 1–40 个字符，不能包含路径或控制字符")
        return self.root / (hashlib.sha256(name.encode("utf-8")).hexdigest() + ".json")

    def load(self, name: str) -> dict:
        path = self.path(name)
        if name in BUILTINS:
            return dict(BUILTINS[name])
        if not path.is_file() or path.stat().st_size > 16384:
            raise ValueError(f"预设不存在或文件过大：{name}")
        data = json.loads(path.read_text(encoding="utf-8"))
        if (
            not isinstance(data, dict)
            or data.get("version") != 1
            or data.get("name") != name
        ):
            raise ValueError("预设格式或名称不匹配")
        return validate_params(data.get("params"))

    def save(self, name: str, params: dict, owner: str):
        """Atomically synchronize a named preset owned by one provider.

        Args:
            name: User-visible preset name, excluding built-in names.
            params: Complete parameter snapshot.
            owner: Provider instance ID authorized to update this preset.
        """
        path = self.path(name)
        if name in BUILTINS:
            raise ValueError("不能覆盖内置预设")
        params = validate_params(params)
        self.root.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if path.stat().st_size > 16384:
                raise ValueError("预设文件过大")
            previous = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(previous, dict) or previous.get("owner") != owner:
                raise ValueError("该预设由另一提供商维护，请使用其他名称")
        elif sum(1 for _ in self.root.glob("*.json")) >= 64:
            raise ValueError("自定义预设最多 64 个")
        data = {"version": 1, "name": name, "owner": owner, "params": params}
        content = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
        temporary = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("w", encoding="utf-8", newline="\r\n") as stream:
                stream.write(content)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


class AudioService:
    def __init__(self, root: Path, temp: Path):
        self.presets = Presets(root)
        self.temp = temp
        self.slot = asyncio.Semaphore(1)
        self.tasks = set()
        self.closed = False

    async def run(self, *args):
        """Run one child process, killing and reaping it on cancellation.

        Args:
            *args: Executable and literal arguments; no shell interpretation.

        Raises:
            RuntimeError: The child process failed.
        """
        kwargs = {}
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        launch = asyncio.create_task(
            asyncio.create_subprocess_exec(
                *map(str, args),
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
                env={
                    **os.environ,
                    "OPENBLAS_NUM_THREADS": "1",
                    "OMP_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                },
                **kwargs,
            )
        )
        child = None
        try:
            child = await asyncio.shield(launch)
            if await child.wait():
                raise RuntimeError("Audio subprocess failed")
        finally:
            # Finish process creation even if cancellation arrived during spawn.
            if child is None:
                child = await launch
            if child.returncode is None:
                try:
                    child.kill()
                except ProcessLookupError:
                    pass
                await child.wait()

    async def process(self, source, params, *, ffmpeg, max_seconds, timeout):
        """Decode, resynthesize and filter a local upstream audio file.

        Args:
            source: Local audio path returned by the upstream TTS provider.
            params: Complete validated voice parameters.
            ffmpeg: FFmpeg executable path or name.
            max_seconds: Maximum accepted audio duration.
            timeout: Deadline including waiting for the shared worker slot.

        Returns:
            A unique WAV path in the host temporary directory.

        Raises:
            RuntimeError: The service is stopped, full, or processing failed.
            ValueError: The source exceeds input limits.
        """
        if self.closed or len(self.tasks) >= 5:
            raise RuntimeError("Audio processor unavailable or queue full")
        task = asyncio.current_task()
        self.tasks.add(task)
        token = "omnikit_tts_" + uuid.uuid4().hex
        decoded = self.temp / f"{token}_in.wav"
        rendered = self.temp / f"{token}_world.wav"
        output = self.temp / f"{token}.wav"
        success = False
        try:
            self.temp.mkdir(parents=True, exist_ok=True)
            async with asyncio.timeout(timeout):
                async with self.slot:
                    path = Path(source).resolve(strict=True)
                    if (
                        not path.is_file()
                        or not 0 < path.stat().st_size <= 50 * 1024**2
                    ):
                        raise ValueError("Upstream audio exceeds size limit")
                    # Restrict decoding to local files and one audio stream.
                    await self.run(
                        ffmpeg,
                        "-nostdin",
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-y",
                        "-protocol_whitelist",
                        "file",
                        "-i",
                        path,
                        "-map",
                        "0:a:0",
                        "-t",
                        max_seconds + 0.1,
                        "-vn",
                        "-ac",
                        "1",
                        "-ar",
                        "24000",
                        "-c:a",
                        "pcm_s16le",
                        decoded,
                    )
                    with wave.open(str(decoded), "rb") as reader:
                        duration = reader.getnframes() / reader.getframerate()
                    if not 0.1 <= duration <= max_seconds:
                        raise ValueError("Upstream audio exceeds duration limit")
                    await self.run(
                        sys.executable,
                        "-B",
                        Path(__file__).with_name("audio.py"),
                        decoded,
                        rendered,
                        json.dumps(params),
                    )
                    filters = (
                        f"atempo={params['speed']},bass=g={params['bass']},"
                        f"treble=g={params['treble']},volume={params['gain']}dB,"
                        "apad=pad_dur=0.005,alimiter=limit=0.97:level=false:attack=5,"
                        "atrim=start=0.005,asetpts=N/SR/TB"
                    )
                    await self.run(
                        ffmpeg,
                        "-nostdin",
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-y",
                        "-i",
                        rendered,
                        "-af",
                        filters,
                        "-c:a",
                        "pcm_s16le",
                        output,
                    )
                    with wave.open(str(output), "rb") as reader:
                        if not reader.getnframes():
                            raise ValueError("Empty processed audio")
                    success = True
                    return str(output)
        finally:
            self.tasks.discard(task)
            decoded.unlink(missing_ok=True)
            rendered.unlink(missing_ok=True)
            if not success:
                output.unlink(missing_ok=True)

    async def close(self):
        self.closed = True
        tasks = tuple(self.tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
