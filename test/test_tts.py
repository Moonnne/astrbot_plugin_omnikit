"""Voice processing, provider delegation and preset persistence contracts."""

import asyncio
import importlib
import os
import shutil
import sys
import wave
from copy import deepcopy
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pytest
from astrbot.core.config.default import DEFAULT_CONFIG
from astrbot.core.platform.astr_message_event import current_message_event
from astrbot.core.provider.manager import ProviderManager
from astrbot.core.provider.provider import TTSProvider
from astrbot.core.provider.register import provider_cls_map

from core.audio import BUILTINS, NEUTRAL, validate_params
from core.tts import AudioService, Presets

if "omnikit_test_plugin" not in sys.modules:
    package = ModuleType("omnikit_test_plugin")
    package.__path__ = [str(Path(__file__).parents[1])]
    sys.modules[package.__name__] = package
adapter = importlib.import_module("omnikit_test_plugin.core.adapter.tts_provider")


class Upstream(TTSProvider):
    async def get_audio(self, text):
        return "upstream.wav"


@pytest.fixture
async def binding(tmp_path, monkeypatch):
    source = Upstream({"type": "test", "id": "source"}, {})
    source.get_audio = AsyncMock(return_value="upstream.wav")
    instances = {"source": source}
    context = SimpleNamespace(get_provider_by_id=instances.get)
    service = AudioService(tmp_path / "data", tmp_path / "temp")
    access = SimpleNamespace(allowed=AsyncMock(return_value=True))
    bound = adapter.Binding(context, service, access, True)
    monkeypatch.setattr(adapter.OmniTTS, "binding", bound)
    yield bound, instances
    await service.close()


def provider(**changes):
    return adapter.OmniTTS(
        {**adapter.TEMPLATE, "上游TTS实例ID": "source", **changes}, {}
    )


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), True, "bad", [], 13])
def test_parameter_validation(bad):
    with pytest.raises(ValueError):
        validate_params({**NEUTRAL, "pitch": bad})


def test_presets_atomic_shared_and_owned(tmp_path):
    presets = Presets(tmp_path)
    first = {**NEUTRAL, "pitch": 3}
    presets.save("肥鱼广播", first, "one")
    assert Presets(tmp_path).load("肥鱼广播") == first
    presets.save("肥鱼广播", {**first, "pitch": 4}, "one")
    assert presets.load("肥鱼广播")["pitch"] == 4
    with pytest.raises(ValueError, match="另一"):
        presets.save("肥鱼广播", first, "two")
    with pytest.raises(ValueError, match="内置"):
        presets.save("GLaDOS", first, "one")
    for name in ("../escape", "a\\b", "bad\n", " " * 3):
        with pytest.raises(ValueError):
            presets.path(name)
    assert not list(presets.root.glob("*.tmp"))
    raw = presets.path("肥鱼广播").read_bytes()
    assert b"\r\n" in raw and b"\n" not in raw.replace(b"\r\n", b"")


async def test_instance_overrides_and_preset_roundtrip(binding):
    one = provider(
        **{
            "id": "one",
            "音高偏移（半音）": "3",
            "基频算法": "dio",
            "保存预设名称": "小鱼",
        }
    )
    await one.initialize()
    two = provider(**{"id": "two", "音频预设": "小鱼"})
    assert one.params == two.params
    three = provider(**{"id": "three"})
    assert three.params == BUILTINS["GLaDOS"]
    assert three.params["pitch"] != one.params["pitch"]
    with pytest.raises(ValueError):
        provider(**{"启用后处理": "yes"})
    with pytest.raises(ValueError):
        provider(**{"音高偏移（半音）": "nan"})


async def test_delegate_and_resolve_current_source(binding, monkeypatch):
    bound, instances = binding
    post = provider()
    process = AsyncMock(return_value="processed.wav")
    monkeypatch.setattr(bound.service, "process", process)
    assert await post.get_audio("文字") == "processed.wav"
    instances["source"].get_audio.assert_awaited_once_with("文字")
    replacement = Upstream({"type": "test"}, {})
    replacement.get_audio = AsyncMock(return_value="replacement.mp3")
    instances["source"] = replacement
    await post.get_audio("下一次")
    assert process.call_args.args[0] == "replacement.mp3"
    assert post.support_stream() is False


async def test_disable_scope_and_reload_passthrough(binding, monkeypatch):
    bound, _ = binding
    process = AsyncMock(return_value="effect.wav")
    monkeypatch.setattr(bound.service, "process", process)
    assert await provider(**{"启用后处理": "false"}).get_audio("test") == "upstream.wav"
    post = provider()
    bound.enabled = False
    assert await post.get_audio("test") == "upstream.wav"
    bound.enabled = True
    token = current_message_event.set(SimpleNamespace())
    bound.adapter.allowed.return_value = False
    try:
        assert await post.get_audio("test") == "upstream.wav"
    finally:
        current_message_event.reset(token)
    process.assert_not_awaited()
    monkeypatch.setattr(adapter.OmniTTS, "binding", None)
    assert await post.get_audio("test") == "upstream.wav"
    replacement = SimpleNamespace(binding=bound)
    monkeypatch.setattr(provider_cls_map[adapter.TYPE], "cls_type", replacement)
    assert await post.get_audio("test") == "effect.wav"


async def test_errors_fallback_health_and_cycle_rejection(binding, monkeypatch):
    bound, instances = binding
    post = provider()
    monkeypatch.setattr(bound.service, "process", AsyncMock(side_effect=TimeoutError))
    assert await post.get_audio("test") == "upstream.wav"
    with pytest.raises(RuntimeError, match="后处理失败"):
        await post.test()
    instances["source"].get_audio.side_effect = OSError("upstream unavailable")
    with pytest.raises(OSError):
        await post.get_audio("test")
    instances["source"] = provider(**{"id": "other"})
    with pytest.raises(ValueError, match="循环"):
        await post.get_audio("test")
    instances.pop("source")
    with pytest.raises(ValueError, match="不存在"):
        await post.get_audio("test")


async def test_health_check_matches_host_probe(binding, tmp_path, monkeypatch):
    bound, instances = binding
    source = instances["source"]
    original = tmp_path / "source.wav"
    original.write_bytes(b"audio")
    source.get_audio.return_value = str(original)
    await TTSProvider.test(source)
    host_text = source.get_audio.call_args.args[0]
    original.write_bytes(b"audio")

    async def synthesize(text):
        if text != host_text:
            raise RuntimeError(
                "音频生成失败: No audio was received. "
                "Please verify that your parameters are correct."
            )
        return str(original)

    source.get_audio.side_effect = synthesize
    output = tmp_path / "processed.wav"
    output.write_bytes(b"processed audio")
    process = AsyncMock(return_value=str(output))
    monkeypatch.setattr(bound.service, "process", process)
    post = provider(**{"音频预设": next(iter(BUILTINS))})
    await post.test()
    source.get_audio.assert_awaited_with(host_text)
    process.assert_awaited_once()
    assert original.is_file()


async def test_host_provider_manager_loads_multiple_instances(binding):
    bound, sources = binding
    manager = ProviderManager(
        SimpleNamespace(confs={"default": deepcopy(DEFAULT_CONFIG)}),
        None,
        SimpleNamespace(default_persona="default"),
    )
    configs = [
        {
            **adapter.TEMPLATE,
            "id": name,
            "enable": True,
            "上游TTS实例ID": "source",
            "保存预设名称": name,
        }
        for name in ("voice_one", "voice_two")
    ]
    for config in configs:
        await manager.load_provider(config)
    assert len(manager.tts_provider_insts) == 2
    assert manager.inst_map["voice_one"].meta().id == "voice_one"
    assert manager.inst_map["voice_two"].meta().id == "voice_two"
    assert bound.service.presets.load("voice_one") == BUILTINS["GLaDOS"]
    assert "source" in sources
    for config in configs:
        await manager.terminate_provider(config["id"])
    assert not manager.inst_map and not manager.tts_provider_insts


async def test_host_provider_schema_exposes_template(binding):
    from astrbot.dashboard.services.config_service import ProviderConfigService

    service = ProviderConfigService(
        SimpleNamespace(astrbot_config={}, provider_manager=None)
    )
    template = service.get_provider_schema()["config_schema"]["provider"][
        "config_template"
    ][adapter.TYPE]
    assert template["provider_type"] == "text_to_speech"
    assert template["音高偏移（半音）"] == ""
    assert template["音频预设"] == "GLaDOS"


@pytest.fixture
def ffmpeg():
    command = os.environ.get("OMNIKIT_TEST_FFMPEG") or shutil.which("ffmpeg")
    if not command:
        pytest.skip(
            "Set OMNIKIT_TEST_FFMPEG or install FFmpeg for audio integration tests"
        )
    return command


@pytest.fixture
def voice(tmp_path):
    rate = 24000
    t = np.arange(rate * 2) / rate
    x = sum(np.sin(2 * np.pi * 180 * k * t) / k for k in range(1, 30)) * 0.25
    x *= np.minimum(1, np.minimum(t, 2 - t) * 20)
    path = tmp_path / "voice.wav"
    with wave.open(str(path), "wb") as out:
        out.setparams((1, 2, rate, 0, "NONE", ""))
        out.writeframes((x * 32767).astype("<i2").tobytes())
    return path


async def test_real_world_pitch_speed_and_source_preservation(tmp_path, ffmpeg, voice):
    import pyworld

    service = AudioService(tmp_path / "data", tmp_path / "temp")
    original = voice.read_bytes()
    params = {**NEUTRAL, "pitch": 12, "speed": 1.25, "method": "dio"}
    output = await service.process(
        voice, params, ffmpeg=ffmpeg, max_seconds=3, timeout=30
    )
    with wave.open(output, "rb") as reader:
        assert reader.getframerate() == 24000
        assert reader.getnchannels() == 1 and reader.getsampwidth() == 2
        x = (
            np.frombuffer(reader.readframes(reader.getnframes()), "<i2").astype(
                np.float64
            )
            / 32768
        )
    f0, _ = pyworld.dio(x, 24000)
    assert np.median(f0[f0 > 0]) == pytest.approx(360, rel=0.05)
    assert len(x) / 24000 == pytest.approx(1.6, abs=0.05)
    assert np.isfinite(x).all() and 0.01 < np.max(np.abs(x)) <= 0.98
    assert voice.read_bytes() == original
    assert list(service.temp.iterdir()) == [Path(output)]
    await service.close()


async def test_real_glados_and_duration_limit(tmp_path, ffmpeg, voice):
    service = AudioService(tmp_path / "data", tmp_path / "temp")
    output = await service.process(
        voice, BUILTINS["GLaDOS"], ffmpeg=ffmpeg, max_seconds=3, timeout=30
    )
    assert Path(output).read_bytes() != voice.read_bytes()
    with pytest.raises(ValueError, match="duration"):
        await service.process(voice, NEUTRAL, ffmpeg=ffmpeg, max_seconds=1, timeout=30)
    assert list(service.temp.iterdir()) == [Path(output)]
    await service.close()


@pytest.mark.parametrize("during_spawn", [False, True])
async def test_subprocess_cancellation_reaps_child(tmp_path, monkeypatch, during_spawn):
    service = AudioService(tmp_path, tmp_path / "temp")
    create = asyncio.create_subprocess_exec
    ready = asyncio.Event()
    release = asyncio.Event()
    processes = []

    async def capture(*args, **kwargs):
        child = await create(*args, **kwargs)
        processes.append(child)
        ready.set()
        if during_spawn:
            await release.wait()
        return child

    monkeypatch.setattr(asyncio, "create_subprocess_exec", capture)
    task = asyncio.create_task(
        service.run(sys.executable, "-c", "import time; time.sleep(30)")
    )
    await asyncio.wait_for(ready.wait(), 5)
    task.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert processes[0].returncode is not None


async def test_queue_limit_and_plugin_close(tmp_path, voice, monkeypatch):
    service = AudioService(tmp_path / "data", tmp_path / "temp")
    blocked = asyncio.Event()

    async def wait(*args):
        await blocked.wait()

    monkeypatch.setattr(service, "run", wait)
    kwargs = dict(ffmpeg="ffmpeg", max_seconds=3, timeout=30)
    jobs = [
        asyncio.create_task(service.process(voice, NEUTRAL, **kwargs)) for _ in range(5)
    ]
    await asyncio.sleep(0)
    assert len(service.tasks) == 5
    with pytest.raises(RuntimeError, match="queue"):
        await service.process(voice, NEUTRAL, **kwargs)
    await service.close()
    assert all(job.cancelled() for job in jobs)
    assert not service.tasks and not list(service.temp.iterdir())
