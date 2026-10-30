"""AstrBot TTS provider registration, delegation and lifecycle binding."""

import asyncio
import importlib.util
from dataclasses import dataclass
from pathlib import Path

from astrbot.core import logger
from astrbot.core.platform.astr_message_event import current_message_event
from astrbot.core.provider.entities import ProviderType
from astrbot.core.provider.provider import TTSProvider
from astrbot.core.provider.register import provider_cls_map, register_provider_adapter

from ..audio import PARAMETERS, number, validate_params
from ..tts import AudioService
from .astrbot import Adapter

TYPE = "omnikit_tts_postprocess"
TEMPLATE = {
    "type": TYPE,
    "provider_type": "text_to_speech",
    "id": "omnikit_tts",
    "enable": False,
    "hint": (
        "选择一个已启用的上游 TTS 实例 ID，再将本实例设为会话的 TTS。"
        "后处理开关填写 true/false；关闭时透传。预设可填 GLaDOS、中性或自定义名称；"
        "下方留空参数继承预设，填数字则覆盖。保存预设填写新名称，"
        "该实例每次保存/重载都会同步当前效果；清空此字段停止同步。"
        "自定义预设保存在 data/plugin_data/astrbot_plugin_omnikit/tts/presets。"
        "调参范围见插件 README；保存提供商配置后生效。"
    ),
    "上游TTS实例ID": "",
    "启用后处理": "true",
    "音频预设": "GLaDOS",
    "保存预设名称": "",
    **{spec[0]: "" for spec in PARAMETERS.values()},
    "基频算法": "",
    "FFmpeg路径": "ffmpeg",
    "音频时长上限（秒）": 60,
    "处理超时（秒）": 60,
}


@dataclass
class Binding:
    context: object
    service: AudioService
    adapter: Adapter
    enabled: bool


class OmniTTS(TTSProvider):
    binding = None
    omnikit_postprocessor = True

    def __init__(self, provider_config: dict, provider_settings: dict):
        super().__init__(provider_config, provider_settings)
        binding = self.current_binding()
        if binding is None:
            raise ValueError("请先启用并加载万能工具箱插件")
        self.context = binding.context
        self.source_id = str(provider_config.get("上游TTS实例ID", "")).strip()
        if not self.source_id or self.source_id == provider_config.get("id"):
            raise ValueError("上游TTS实例ID须填写其他已启用的 TTS 实例")
        switch = str(provider_config.get("启用后处理", "true")).lower().strip()
        if switch not in {"true", "false"}:
            raise ValueError("启用后处理须为 true 或 false")
        self.effect_enabled = switch == "true"
        self.ffmpeg = str(provider_config.get("FFmpeg路径", "ffmpeg")).strip()
        if not self.ffmpeg:
            raise ValueError("FFmpeg路径不能为空")
        self.max_seconds = number(
            provider_config.get("音频时长上限（秒）", 60), "音频时长上限", 1, 180
        )
        self.timeout = number(
            provider_config.get("处理超时（秒）", 60), "处理超时", 5, 300
        )
        self.params = binding.service.presets.load(
            str(provider_config.get("音频预设", "GLaDOS")).strip()
        )
        for key, (label, _, _, _) in PARAMETERS.items():
            value = provider_config.get(label, "")
            if value != "" and value is not None:
                self.params[key] = value
        method = str(provider_config.get("基频算法", "")).strip()
        if method:
            self.params["method"] = method
        self.params = validate_params(self.params)
        self.save_as = str(provider_config.get("保存预设名称", "")).strip()
        self.jobs = set()
        self.closed = False
        self.set_model("WORLD")

    @staticmethod
    def current_binding():
        # Registry indirection lets existing instances follow plugin reloads.
        metadata = provider_cls_map.get(TYPE)
        return getattr(metadata.cls_type, "binding", None) if metadata else None

    async def initialize(self):
        binding = self.current_binding()
        if binding is None:
            raise ValueError("万能工具箱插件未加载")
        if self.effect_enabled:
            for module in ("numpy", "pyworld"):
                if importlib.util.find_spec(module) is None:
                    raise ValueError(
                        f"缺少 {module}，请在 AstrBot 环境安装插件 requirements.txt"
                    )
        if self.save_as:
            binding.service.presets.save(self.save_as, self.params, self.meta().id)

    async def get_audio(self, text: str) -> str:
        return await self.audio(text, strict=False)

    async def audio(self, text: str, *, strict: bool) -> str:
        """Delegate synthesis and apply the current plugin's audio service.

        Args:
            text: Text to forward unchanged to the upstream TTS instance.
            strict: Raise processing errors during provider health checks.

        Returns:
            Processed WAV path, or upstream path when processing is bypassed.
        """
        if self.closed:
            raise RuntimeError("TTS provider is closed")
        binding = self.current_binding()
        context = binding.context if binding else self.context
        source = context.get_provider_by_id(self.source_id)
        if not isinstance(source, TTSProvider):
            raise ValueError("上游 TTS 不存在、未启用或类型错误")
        if source is self or getattr(source, "omnikit_postprocessor", False):
            raise ValueError("上游不能选择 TTS 后处理提供商，避免循环或重复处理")
        task = asyncio.current_task()
        self.jobs.add(task)
        try:
            original = await source.get_audio(text)
            if not self.effect_enabled or binding is None or not binding.enabled:
                return original
            try:
                event = current_message_event.get()
                if event is not None and not await binding.adapter.allowed(event):
                    return original
                return await binding.service.process(
                    original,
                    self.params,
                    ffmpeg=self.ffmpeg,
                    max_seconds=self.max_seconds,
                    timeout=self.timeout,
                )
            except Exception as exc:
                if strict:
                    raise RuntimeError(
                        f"TTS 后处理失败：{type(exc).__name__}"
                    ) from None
                logger.warning(
                    "OmniKit TTS processing failed (%s); using upstream audio",
                    type(exc).__name__,
                )
                return original
        finally:
            self.jobs.discard(task)

    async def test(self):
        output = await self.audio("hi", strict=True)
        if not Path(output).is_file() or Path(output).stat().st_size == 0:
            raise ValueError("TTS 没有生成有效的音频文件")

    async def terminate(self):
        self.closed = True
        jobs = tuple(self.jobs)
        for job in jobs:
            job.cancel()
        if jobs:
            await asyncio.gather(*jobs, return_exceptions=True)


existing = provider_cls_map.get(TYPE)
if existing is None:
    register_provider_adapter(
        TYPE,
        "WORLD 语音后处理，可调用其他 TTS 实例",
        provider_type=ProviderType.TEXT_TO_SPEECH,
        default_config_tmpl=TEMPLATE,
        provider_display_name="万能工具箱 · TTS 后处理",
    )(OmniTTS)
elif getattr(existing.cls_type, "omnikit_postprocessor", False):
    existing.cls_type = OmniTTS
    existing.default_config_tmpl = TEMPLATE
else:
    raise ValueError("OmniKit TTS provider type is already registered")
