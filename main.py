"""AstrBot registration and lifecycle for the whale toolbox."""

from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star

from .core.adapter.astrbot import Adapter, PendingImageFilter
from .core.config import Settings
from .core.gateway import Gateway


class OmniKit(Star):
    def __init__(self, context: Context, config=None):
        super().__init__(context)
        self.gateway = Gateway(
            Settings.parse(config or {}), Adapter(context), self.logger
        )
        PendingImageFilter.gateways.add(self.gateway)

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
        PendingImageFilter.gateways.discard(self.gateway)
        await self.gateway.close()
