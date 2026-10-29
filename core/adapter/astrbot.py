"""Keep AstrBot message, media and access details at the boundary."""

import asyncio
import base64
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname
from weakref import WeakSet

import aiohttp
from astrbot.api.event import filter
from astrbot.api.message_components import Image
from astrbot.core.star.filter.command import CommandFilter
from astrbot.core.star.filter.command_group import CommandGroupFilter
from astrbot.core.star.filter.regex import RegexFilter
from astrbot.core.star.session_plugin_manager import SessionPluginManager

from ..errors import UserError

PLUGIN = "astrbot_plugin_omnikit"


class PendingImageFilter(filter.CustomFilter):
    gateways = WeakSet()

    def filter(self, event, cfg):
        return any(g.accepts_follow(event) for g in tuple(self.gateways))


class Adapter:
    def __init__(self, context):
        self.context = context

    def key(self, event) -> tuple[str, str, str, str]:
        account = str(event.get_self_id())
        if event.get_platform_name() in {"qq_official", "qq_official_webhook"}:
            platform = self.context.get_platform_inst(event.get_platform_id())
            appid = getattr(platform, "appid", None)
            if appid:
                account = str(appid)
        return (
            str(event.get_platform_id()),
            account,
            str(event.unified_msg_origin),
            str(event.get_sender_id()),
        )

    async def allowed(self, event) -> bool:
        if str(event.get_sender_id()) == str(event.get_self_id()):
            return False
        cfg = self.context.get_config(event.unified_msg_origin)
        selected = cfg.get("plugin_set", ["*"])
        if selected != ["*"] and PLUGIN not in selected:
            return False
        platform = cfg.get("platform_settings", {})
        whitelist = set(map(str, platform.get("id_whitelist", [])))
        if platform.get("enable_id_white_list") and whitelist:
            candidates = {
                str(event.unified_msg_origin),
                str(event.get_group_id() or event.get_sender_id()),
            }
            if not candidates & whitelist:
                return False
        return await SessionPluginManager.is_plugin_enabled_for_session(
            event.unified_msg_origin, PLUGIN
        )

    @staticmethod
    def images(event) -> list:
        return [part for part in event.get_messages() if isinstance(part, Image)]

    @staticmethod
    def arguments(event) -> str:
        # WakingCheckStage removes the configured prefix, preserving inner whitespace.
        for i, char in enumerate(event.message_str):
            if char.isspace():
                return event.message_str[i:].lstrip()
        return ""

    @staticmethod
    def has_command(event) -> bool:
        return any(
            isinstance(f, (CommandFilter, CommandGroupFilter, RegexFilter))
            for h in event.get_extra("activated_handlers", []) or []
            for f in h.event_filters
        )

    @staticmethod
    async def text(event, value: str):
        await event.send(event.plain_result(value).use_markdown(False))

    @staticmethod
    async def picture(event, value: bytes):
        await event.send(event.chain_result([Image.fromBytes(value)]))

    @staticmethod
    async def read(image, limit: int) -> bytes:
        """Read only a host-provided image reference, with a bounded download."""
        ref = image.url or image.file or image.path
        if not ref:
            raise UserError("图片消息没有可读取的附件，请重新发送原图。")
        if ref.startswith(("https://", "http://")):
            timeout = aiohttp.ClientTimeout(total=20)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(ref, max_redirects=3) as response:
                    response.raise_for_status()
                    if response.content_length and response.content_length > limit:
                        raise UserError("图片文件过大。")
                    result = bytearray()
                    async for chunk in response.content.iter_chunked(65536):
                        result.extend(chunk)
                        if len(result) > limit:
                            raise UserError("图片文件过大。")
                    return bytes(result)
        if ref.startswith("base64://"):
            encoded = ref[9:]
            if len(encoded) > 4 * ((limit + 2) // 3):
                raise UserError("图片文件过大。")
            try:
                data = base64.b64decode(encoded, validate=True)
            except ValueError as exc:
                raise UserError("图片附件编码无效。") from exc
        else:
            if ref.startswith("file://"):
                parsed = urlparse(ref)
                if parsed.netloc not in ("", "localhost"):
                    raise UserError("不支持网络共享路径图片。")
                ref = url2pathname(parsed.path)
            path = Path(ref)
            if not path.is_file():
                raise UserError("图片附件已失效，请重新发送。")

            def read_file():
                with path.open("rb") as stream:
                    return stream.read(limit + 1)

            data = await asyncio.to_thread(read_file)
        if len(data) > limit:
            raise UserError("图片文件过大。")
        return data
