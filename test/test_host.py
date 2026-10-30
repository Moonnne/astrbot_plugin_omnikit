"""Real host filters, pipeline and platform message contracts; no network sends."""

import asyncio
import importlib
import sys
from copy import deepcopy
from pathlib import Path
from types import MethodType, ModuleType, SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from aiocqhttp import Event as CQEvent
from astrbot.api.message_components import Image, Plain
from astrbot.api.platform import (
    AstrBotMessage,
    MessageMember,
    MessageType,
    PlatformMetadata,
)
from astrbot.core.config.default import DEFAULT_CONFIG
from astrbot.core.pipeline.process_stage.method.star_request import StarRequestSubStage
from astrbot.core.pipeline.waking_check.stage import WakingCheckStage
from astrbot.core.platform.sources.aiocqhttp.aiocqhttp_message_event import (
    AiocqhttpMessageEvent,
)
from astrbot.core.platform.sources.aiocqhttp.aiocqhttp_platform_adapter import (
    AiocqhttpAdapter,
)
from astrbot.core.platform.sources.qqofficial.qqofficial_message_event import (
    QQOfficialMessageEvent,
)
from astrbot.core.platform.sources.qqofficial.qqofficial_platform_adapter import (
    PatchedGroupMessage,
    QQOfficialPlatformAdapter,
)
from astrbot.core.platform.sources.qqofficial_webhook.qo_webhook_event import (
    QQOfficialWebhookMessageEvent,
)
from astrbot.core.star.session_plugin_manager import SessionPluginManager
from astrbot.core.star.star import star_map
from test_codecs import png

from core.text import decode

package = ModuleType("omnikit_test_plugin")
package.__path__ = [str(Path(__file__).parents[1])]
sys.modules[package.__name__] = package
main = importlib.import_module("omnikit_test_plugin.main")

CLASSES = {
    "aiocqhttp": AiocqhttpMessageEvent,
    "qq_official": QQOfficialMessageEvent,
    "qq_official_webhook": QQOfficialWebhookMessageEvent,
}


def event(text, kind="aiocqhttp", mid="1", images=(), user="100", private=False):
    abm = AstrBotMessage()
    abm.type = MessageType.FRIEND_MESSAGE if private else MessageType.GROUP_MESSAGE
    abm.self_id, abm.group_id = "bot", "" if private else "group"
    abm.session_id = user if private else "group"
    abm.sender = MessageMember(user_id=user, nickname="Fish")
    abm.message_str, abm.message_id = text, mid
    abm.message = [Plain(text)] + list(images)
    meta = PlatformMetadata(name=kind, id=kind, description="test")
    result = CLASSES[kind](text, abm, meta, abm.session_id, SimpleNamespace())
    result.send = AsyncMock()
    return result


@pytest.fixture
async def plugin(monkeypatch):
    cfg = deepcopy(DEFAULT_CONFIG)
    cfg.update(wake_prefix=["/"], admins_id=[], plugin_set=["astrbot_plugin_omnikit"])
    metadata = star_map[main.OmniKit.__module__]
    metadata.name, metadata.activated = "astrbot_plugin_omnikit", True
    context = SimpleNamespace(
        get_config=lambda *args: cfg,
        get_platform_inst=lambda *args: SimpleNamespace(appid="stable-app"),
    )
    monkeypatch.setattr(
        SessionPluginManager,
        "is_plugin_enabled_for_session",
        AsyncMock(return_value=True),
    )
    instance = main.OmniKit(context)
    yield instance, cfg
    await instance.terminate()


async def dispatch(plugin, cfg, message):
    waking = WakingCheckStage()
    await waking.initialize(
        SimpleNamespace(astrbot_config=cfg, db_helper=None, astrbot_config_id="test")
    )
    await waking.process(message)
    originals = []
    try:
        for handler in message.get_extra("activated_handlers", []):
            if handler.handler_module_path == main.OmniKit.__module__:
                originals.append((handler, handler.handler))
                handler.handler = MethodType(handler.handler, plugin)
        request = StarRequestSubStage()
        await request.initialize(SimpleNamespace(astrbot_config=cfg))
        async for _ in request.process(message):
            pass
    finally:
        for handler, original in originals:
            handler.handler = original


@pytest.mark.parametrize("kind", CLASSES)
@pytest.mark.parametrize("private", [True, False])
async def test_real_commands_and_next_image(plugin, kind, private):
    instance, cfg = plugin
    text = event("/鲸言加密 -- hello  world\n鲸鱼", kind, private=private)
    await dispatch(instance, cfg, text)
    assert text.send.await_count == 1
    cipher = text.send.call_args.args[0].chain[0].text
    assert decode(cipher) == "hello  world\n鲸鱼"
    assert not text.call_llm
    wait = event("/鲸绘解混 3", kind, mid="2", private=private)
    await dispatch(instance, cfg, wait)
    assert "60 秒" in wait.send.call_args.args[0].chain[0].text
    unrelated = event(
        "",
        kind,
        mid="3",
        images=[Image.fromBytes(png())],
        user="other",
        private=private,
    )
    await dispatch(instance, cfg, unrelated)
    assert not unrelated.send.called
    picture = event("", kind, mid="4", images=[Image.fromBytes(png())], private=private)
    await dispatch(instance, cfg, picture)
    assert picture.send.await_count == 1
    image = picture.send.call_args.args[0].chain[0]
    assert isinstance(image, Image)
    assert (await instance.gateway.api.read(image, 10000)).startswith(b"\x89PNG")
    assert not instance.gateway.pending


@pytest.mark.parametrize("command", ["鲸言", "肥鱼嘀咕", "鲸绘", "鲸绘取消"])
async def test_help_and_cancel(plugin, command):
    instance, cfg = plugin
    message = event("/" + command)
    await dispatch(instance, cfg, message)
    assert message.send.await_count == 1


async def test_prefix_plain_chat_and_disabled_session(plugin):
    instance, cfg = plugin
    plain = event("鲸绘加混")
    await dispatch(instance, cfg, plain)
    assert not plain.send.called and not instance.gateway.pending
    cfg["wake_prefix"] = ["!"]
    message = event("!肥鱼嘀咕 我是肥鱼 你好", mid="2")
    await dispatch(instance, cfg, message)
    assert decode(message.send.call_args.args[0].chain[0].text) == "你好"
    SessionPluginManager.is_plugin_enabled_for_session.return_value = False
    disabled = event("!鲸绘加混", mid="3")
    await dispatch(instance, cfg, disabled)
    assert not disabled.send.called and not instance.gateway.pending


async def test_official_attachment_normalization():
    raw = PatchedGroupMessage(
        None,
        "event",
        {
            "id": "message",
            "content": "/鲸绘加混 3",
            "group_openid": "group",
            "author": {"member_openid": "user"},
            "attachments": [
                {
                    "content_type": "image/png",
                    "url": "cdn.example/image.png",
                    "filename": "image.png",
                }
            ],
        },
    )
    abm = await QQOfficialPlatformAdapter._parse_from_qqofficial(
        raw, MessageType.GROUP_MESSAGE, True
    )
    assert abm.message_str.strip() == "/鲸绘加混 3"
    images = [part for part in abm.message if isinstance(part, Image)]
    assert len(images) == 1 and images[0].file == "https://cdn.example/image.png"


async def test_onebot_attachment_normalization():
    platform = AiocqhttpAdapter(
        {"id": "onebot", "ws_reverse_host": "127.0.0.1", "ws_reverse_port": 0},
        {},
        asyncio.Queue(),
    )
    raw = CQEvent(
        {
            "time": 1,
            "self_id": 999,
            "post_type": "message",
            "message_type": "group",
            "sub_type": "normal",
            "message_id": 1,
            "group_id": 123,
            "user_id": 456,
            "sender": {"user_id": 456, "nickname": "Fish", "role": "member"},
            "message": [
                {"type": "text", "data": {"text": "/鲸绘加混 3"}},
                {
                    "type": "image",
                    "data": {
                        "file": "image.png",
                        "url": "https://cdn.example/image.png",
                    },
                },
            ],
            "raw_message": "/鲸绘加混 3",
        }
    )
    abm = await platform.convert_message(raw)
    assert abm.message_str.strip() == "/鲸绘加混 3"
    assert len([p for p in abm.message if isinstance(p, Image)]) == 1


async def test_media_read_local_base64_and_limits(tmp_path):
    adapter = main.Adapter(SimpleNamespace())
    raw = png()
    path = tmp_path / "fish.png"
    path.write_bytes(raw)
    for part in (Image.fromFileSystem(path), Image.fromBytes(raw)):
        assert await adapter.read(part, len(raw)) == raw
        with pytest.raises(ValueError):
            await adapter.read(part, 1)
