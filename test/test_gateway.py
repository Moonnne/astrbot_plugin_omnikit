"""Concurrency, follow-up ownership and lifecycle checks."""

import asyncio
import logging
from dataclasses import replace
from threading import Event as ThreadEvent
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from test_codecs import png

from core.config import Settings
from core.gateway import Gateway, Pending
from core.text import decode


class Event:
    def __init__(
        self, text="", user="u", room="r", platform="p", account="a", images=(), mid="1"
    ):
        self.message_str = text
        self.key = (platform, account, room, user)
        self.images = list(images)
        self.message_obj = SimpleNamespace(message_id=mid)
        self.stopped = False
        self.llm = True
        self.command = False

    def get_sender_id(self):
        return self.key[-1]

    def get_self_id(self):
        return self.key[1]

    def get_platform_name(self):
        return self.key[0]

    def should_call_llm(self, value):
        self.llm = value

    def stop_event(self):
        self.stopped = True


def gateway(**settings):
    api = SimpleNamespace(
        key=lambda e: e.key,
        allowed=AsyncMock(return_value=True),
        images=lambda e: e.images,
        arguments=lambda e: e.message_str.partition(" ")[2],
        has_command=lambda e: e.command,
        text=AsyncMock(),
        picture=AsyncMock(),
        read=AsyncMock(return_value=png()),
    )
    return Gateway(replace(Settings(), **settings), api, logging.getLogger("test"))


async def test_wait_owner_expiry_replacement_and_cancel(monkeypatch):
    now = [10.0]
    monkeypatch.setattr("core.gateway.monotonic", lambda: now[0])
    g = gateway()
    initial = Event("鲸绘解混 3")
    await g.handle(initial, "restore")
    assert g.pending[initial.key] == Pending(True, 3, 70.0)
    assert initial.stopped and not initial.llm
    for kwargs in (
        {"user": "v"},
        {"room": "other"},
        {"platform": "q"},
        {"account": "b"},
    ):
        assert not g.accepts_follow(Event(images=[1], **kwargs))
    assert not g.accepts_follow(Event())
    assert g.accepts_follow(Event(images=[1]))
    now[0] = 69.9
    assert g.accepts_follow(Event(images=[1]))
    now[0] = 70
    assert not g.accepts_follow(Event(images=[1]))
    await g.handle(Event("鲸绘加混", mid="2"), "scramble")
    await g.handle(Event("鲸绘解混 2", mid="3"), "restore")
    assert g.pending[initial.key].rounds == 2
    await g.handle(Event("鲸绘取消", mid="4"), "cancel")
    assert not g.pending
    await g.close()


async def test_followup_atomic_claim_and_duplicate():
    g = gateway()
    await g.handle(Event("鲸绘加混"), "scramble")
    a, b = Event(images=[1], mid="2"), Event(images=[1], mid="3")
    await asyncio.gather(g.handle(a, "follow"), g.handle(b, "follow"))
    assert g.api.picture.await_count == 1
    await g.handle(Event(images=[1], mid="2"), "scramble")
    assert g.api.picture.await_count == 1
    assert not g.pending
    await g.close()


async def test_access_change_command_priority_and_multi_images():
    g = gateway()
    await g.handle(Event("鲸绘加混"), "scramble")
    command = Event(images=[1], mid="2")
    command.command = True
    await g.handle(command, "follow")
    assert not command.stopped and g.pending
    await g.handle(Event(images=[1, 2], mid="3"), "follow")
    assert g.pending and not g.api.picture.called
    g.api.allowed.return_value = False
    await g.handle(Event(images=[1], mid="4"), "follow")
    assert not g.api.picture.called
    await g.close()


async def test_inline_image_rounds_overload_and_no_llm():
    g = gateway(workers=1, queue_size=0)
    gate = asyncio.Event()

    async def read(*args):
        await gate.wait()
        return png()

    g.api.read.side_effect = read
    first = asyncio.create_task(g.handle(Event("鲸绘加混 3", images=[1]), "scramble"))
    while not g.api.read.called:
        await asyncio.sleep(0)
    await g.handle(Event("鲸绘加混", user="v", images=[1]), "scramble")
    assert "厨房" in g.api.text.call_args.args[1]
    gate.set()
    await first
    assert g.api.picture.await_count == 1 and not g.busy and g.admitted == 0
    await g.close()


async def test_failed_prompt_does_not_arm_wait_and_text_errors():
    g = gateway()
    g.api.text.side_effect = RuntimeError("delivery")
    await g.handle(Event("鲸绘加混"), "scramble")
    assert not g.pending
    assert (
        decode(g.text_action("encode", "-- hello  world\n鲸鱼")) == "hello  world\n鲸鱼"
    )
    assert decode(g.text_action("encode", "我是肥鱼 你好")) == "你好"
    for value in ("1.1", "-1", "0", "11", "abc"):
        g.api.text.side_effect = None
        await g.handle(Event("鲸绘加混 " + value, mid=value), "scramble")
        assert not g.pending
    await g.close()


async def test_unload_cancels_download_and_waits():
    g = gateway()
    gate = asyncio.Event()
    g.api.read.side_effect = lambda *args: None

    async def read(*args):
        await gate.wait()

    g.api.read.side_effect = read
    task = asyncio.create_task(g.handle(Event("鲸绘加混", images=[1]), "scramble"))
    while not g.api.read.called:
        await asyncio.sleep(0)
    await g.close()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not g.api.picture.called and not g.busy and g.admitted == 0


async def test_prompt_capacity_and_cancel_during_delivery():
    g = gateway(pending_limit=1)
    delivering = asyncio.Event()
    release = asyncio.Event()

    async def send(event, text):
        if "准备好" in text:
            delivering.set()
            await release.wait()

    g.api.text.side_effect = send
    first = asyncio.create_task(g.handle(Event("鲸绘加混"), "scramble"))
    await delivering.wait()
    await g.handle(Event("鲸绘加混", user="v"), "scramble")
    assert len(g.prompts) == 1 and not g.pending
    await g.handle(Event("鲸绘取消", mid="2"), "cancel")
    release.set()
    await first
    assert not g.pending and not g.prompts
    await g.close()


async def test_timeout_holds_slot_until_worker_really_exits(monkeypatch):
    g = gateway(task_seconds=0.05, workers=1, queue_size=0)
    started, release, cancelled = ThreadEvent(), ThreadEvent(), ThreadEvent()

    def slow(raw, rounds, reverse, cfg, stop):
        started.set()
        stop.wait(2)
        cancelled.set()
        release.wait(2)
        return png()

    monkeypatch.setattr("core.gateway.transform", slow)
    task = asyncio.create_task(g.handle(Event("鲸绘加混", images=[1]), "scramble"))
    try:
        assert await asyncio.to_thread(started.wait, 2)
        assert await asyncio.to_thread(cancelled.wait, 2)
        assert g.admitted == 1 and not task.done()
        await g.handle(Event("鲸绘加混", user="v", images=[1]), "scramble")
        assert "厨房" in g.api.text.call_args.args[1]
    finally:
        release.set()
        await task
        await g.close()
    assert not g.api.picture.called and not g.busy and g.admitted == 0


async def test_host_valueerror_does_not_leak_details():
    g = gateway()
    g.api.picture.side_effect = ValueError("private media URL and token")
    await g.handle(Event("鲸绘加混", images=[1]), "scramble")
    assert "private" not in g.api.text.call_args.args[1]
    await g.close()
