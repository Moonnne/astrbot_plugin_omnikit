"""AstrBot registration and lifecycle for the whale toolbox."""

from pathlib import Path

from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, StarTools
from astrbot.core.utils.astrbot_path import get_astrbot_temp_path

from .core.adapter.astrbot import Adapter, PendingImageFilter
from .core.adapter.tts_provider import Binding, OmniTTS
from .core.config import Settings
from .core.gateway import Gateway
from .core.tts import AudioService


class OmniKit(Star):
    def __init__(self, context: Context, config=None):
        super().__init__(context)
        self.gateway = Gateway(
            Settings.parse(config or {}), Adapter(context), self.logger
        )
        PendingImageFilter.gateways.add(self.gateway)
        self.tts_binding = Binding(
            context,
            AudioService(
                StarTools.get_data_dir("astrbot_plugin_omnikit"),
                Path(get_astrbot_temp_path()),
            ),
            self.gateway.api,
            self.gateway.cfg.enabled,
        )
        OmniTTS.binding = self.tts_binding

    async def initialize(self):
        """Refresh provider instances retained across a plugin reload."""
        manager = self.context.provider_manager
        selected = manager.curr_tts_provider_inst
        selected_id = selected.provider_config.get("id") if selected else None
        try:
            for instance in tuple(manager.tts_provider_insts):
                if (
                    getattr(instance, "omnikit_postprocessor", False)
                    and type(instance) is not OmniTTS
                ):
                    await manager.reload(dict(instance.provider_config))
        finally:
            if selected_id in manager.inst_map:
                manager.curr_tts_provider_inst = manager.inst_map[selected_id]

    @filter.command("鲸言")
    async def text_help(self, event: AstrMessageEvent):
        await self.gateway.handle(event, "text_help")

    @filter.command("鲸言加密", alias={"肥鱼嘀咕"})
    async def encode(self, event: AstrMessageEvent):
        await self.gateway.handle(event, "encode")

    @filter.command("鲸言解密", alias={"肥鱼翻译"})
    async def decode(self, event: AstrMessageEvent):
        await self.gateway.handle(event, "decode")

    @filter.command("鲸绘")
    async def image_help(self, event: AstrMessageEvent):
        await self.gateway.handle(event, "image_help")

    @filter.command("鲸绘加混", alias={"肥鱼拌饭"})
    async def scramble(self, event: AstrMessageEvent):
        await self.gateway.handle(event, "scramble")

    @filter.command("鲸绘解混", alias={"肥鱼摆盘"})
    async def restore(self, event: AstrMessageEvent):
        await self.gateway.handle(event, "restore")

    @filter.command("鲸绘取消")
    async def cancel(self, event: AstrMessageEvent):
        await self.gateway.handle(event, "cancel")

    @filter.custom_filter(PendingImageFilter, False, priority=-10)
    async def follow_image(self, event: AstrMessageEvent):
        await self.gateway.handle(event, "follow")

    async def terminate(self):
        if OmniTTS.current_binding() is self.tts_binding:
            OmniTTS.binding = None
        await self.tts_binding.service.close()
        PendingImageFilter.gateways.discard(self.gateway)
        await self.gateway.close()
