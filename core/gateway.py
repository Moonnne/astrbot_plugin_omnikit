"""Command routing and bounded, user-scoped image conversations."""

import asyncio
import re
from collections import OrderedDict
from dataclasses import dataclass
from threading import Event
from time import monotonic

from .config import Settings
from .errors import UserError
from .image import transform
from .text import decode, encode, parse_text


@dataclass(frozen=True)
class Pending:
    reverse: bool
    rounds: int
    deadline: float


class Gateway:
    def __init__(self, settings: Settings, adapter, logger):
        self.cfg, self.api, self.log = settings, adapter, logger
        self.pending = {}
        self.prompts = {}
        self.busy = set()
        self.seen = OrderedDict()
        self.tasks = set()
        self.stops = set()
        self.slots = asyncio.Semaphore(settings.workers)
        self.admitted = 0
        self.closed = False

    def prune(self):
        now = monotonic()
        for key in [
            key for key, value in self.pending.items() if value.deadline <= now
        ]:
            self.pending.pop(key, None)
        while self.seen and next(iter(self.seen.values())) <= now:
            self.seen.popitem(last=False)

    def accepts_follow(self, event) -> bool:
        self.prune()
        return (
            not self.closed
            and self.cfg.enabled
            and str(event.get_sender_id()) != str(event.get_self_id())
            and self.api.key(event) in self.pending
            and bool(self.api.images(event))
        )

    async def handle(self, event, action: str):
        if self.closed or not self.cfg.enabled or not await self.api.allowed(event):
            return
        if self.closed:
            return
        if action == "follow" and (
            not self.accepts_follow(event) or self.api.has_command(event)
        ):
            return
        self.prune()
        key = self.api.key(event)
        message_id = str(event.message_obj.message_id or "")
        receipt = (key, message_id)
        if message_id and receipt in self.seen:
            event.should_call_llm(False)
            event.stop_event()
            return
        if message_id:
            self.seen[receipt] = monotonic() + 300
            if len(self.seen) > 4096:
                self.seen.popitem(last=False)
        event.should_call_llm(False)
        event.stop_event()
        task = asyncio.create_task(self.execute(event, action, key))
        self.tasks.add(task)
        try:
            await task
        finally:
            self.tasks.discard(task)

    async def execute(self, event, action, key):
        try:
            args = self.api.arguments(event)
            if action == "text_help":
                await self.api.text(event, self.text_help())
            elif action == "image_help":
                await self.api.text(event, self.image_help())
            elif action in {"encode", "decode"}:
                await self.api.text(event, self.text_action(action, args))
            elif action == "cancel":
                removed = self.pending.pop(key, None)
                removed = self.prompts.pop(key, None) or removed
                await self.api.text(
                    event,
                    "已取消等图。" if removed else "本鲸没在等你的图片",
                )
            elif key in self.busy:
                await self.api.text(event, "本鲸正在处理你的图片，急什么？！")
            else:
                self.busy.add(key)
                try:
                    await self.image_action(event, action, args, key)
                finally:
                    self.busy.discard(key)
        except UserError as exc:
            await self.error(event, str(exc))
        except TimeoutError:
            await self.error(event, "本鲸处理超时了，稍后用较小的图片再试吧")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # Never log message contents, media URLs or platform credentials.
            self.log.warning("OmniKit operation failed: %s", type(exc).__name__)
            await self.error(
                event, "本鲸没能完成操作，附件可能失效或平台发送失败，请重新发送"
            )

    async def error(self, event, message: str):
        try:
            await self.api.text(event, message)
        except Exception as exc:
            self.log.warning("OmniKit error delivery failed: %s", type(exc).__name__)

    def text_action(self, action, args):
        if len(args) > self.cfg.max_output + 32:
            raise UserError("输入文字过长")
        alphabet, body = parse_text(args, self.cfg.presets)
        if action == "encode":
            if len(body) > self.cfg.max_text:
                raise UserError(f"原文最多 {self.cfg.max_text} 字符")
            result = encode(body, alphabet or self.cfg.alphabet)
        else:
            result = decode(body, alphabet)
        if len(result) > self.cfg.max_output:
            raise UserError("结果超过消息长度预算，请缩短原文后重试")
        return result

    async def image_action(self, event, action, args, key):
        pictures = self.api.images(event)
        if len(pictures) > 1:
            raise UserError("本鲸一次只拌一碗饭，只能发送一张图片哦")
        if action == "follow":
            request = self.pending.pop(key, None)
            if request is None or request.deadline <= monotonic():
                return
            reverse, rounds = request.reverse, request.rounds
        else:
            if args and not re.fullmatch(r"[0-9]{1,3}", args):
                raise UserError("操作次数须为正整数，例如 /鲸绘加混 3 [图片]")
            rounds = int(args) if args else 1
            if not 1 <= rounds <= self.cfg.max_rounds:
                raise UserError(f"操作次数须在 1–{self.cfg.max_rounds} 之间")
            reverse = action == "restore"
            self.pending.pop(key, None)
        if not pictures:
            if len(self.pending) + len(self.prompts) >= self.cfg.pending_limit:
                raise UserError("本鲸等图的饭碗满了，稍后再来吧")
            hint = (
                ""
                if event.get_platform_name() in {"qq_official", "qq_official_webhook"}
                else ""
            )
            token = object()
            self.prompts[key] = token
            try:
                await self.api.text(
                    event,
                    f"本鲸准备好{'摆盘' if reverse else '拌饭'}了！\n"
                    f"请在 {self.cfg.wait_seconds} 秒内于本会话发送一张图片{hint}\n"
                    f"超时了就不管你了！使用 /鲸绘取消 可取消本次操作",
                )
                if not self.closed and self.prompts.get(key) is token:
                    self.pending[key] = Pending(
                        reverse, rounds, monotonic() + self.cfg.wait_seconds
                    )
            finally:
                self.prompts.pop(key, None)
            return
        if self.admitted >= self.cfg.workers + self.cfg.queue_size:
            raise UserError("本鲸的厨房忙不过来了，请稍后重试。")
        self.admitted += 1
        try:
            async with asyncio.timeout(self.cfg.task_seconds):
                async with self.slots:
                    raw = await self.api.read(pictures[0], self.cfg.max_bytes)
                    output = await self.process(raw, rounds, reverse)
                    if not self.closed and await self.api.allowed(event):
                        await self.api.picture(event, output)
        finally:
            self.admitted -= 1

    async def process(self, raw, rounds, reverse):
        stop = Event()
        self.stops.add(stop)
        job = asyncio.create_task(
            asyncio.to_thread(transform, raw, rounds, reverse, self.cfg, stop)
        )
        try:
            return await asyncio.shield(job)
        finally:
            stop.set()
            # Keep the worker slot until the actual thread has stopped.
            cancelled = False
            while not job.done():
                try:
                    await asyncio.shield(job)
                except asyncio.CancelledError:
                    cancelled = True
                except Exception:
                    break
            if not job.cancelled():
                job.exception()
            self.stops.discard(stop)
            if cancelled:
                raise asyncio.CancelledError

    def text_help(self):
        return (
            f"本鲸说两句 · 鲸言\n"
            f"使用方式："
            f"/鲸言加密 [正文] 或 /肥鱼嘀咕 *\n"
            f"/鲸言解密 [密文] 或 /肥鱼翻译 *\n"
            f"可用参数："
            f"/鲸言加密 [四字密码] [正文]\n"
            f"/鲸言解密 [四字密码] [密文]\n"
            f"/鲸言加密 兽音 正文\n"
            f"/鲸言加密 -- hello world\n"
            f"默认加密词：{self.cfg.alphabet}"
            f"可用预设：{'、'.join(self.cfg.presets) or '无'}\n\n"
            f"加密词仅支持四个互不重复的可见字符，可使用预设\n"
            f"无参数解密时将尝试从密文推导字表\n"
            f"⚠本功能仅为趣味功能，不具备密码学安全性！"
        )

    def image_help(self):
        return (
            f"肥鱼拌饭机 · 鲸绘\n"
            f"使用方式："
            f"/鲸绘加混 [操作次数] [图片] 或 /肥鱼拌饭 *\n"
            f"/鲸绘解混 [操作次数] [图片] 或 /肥鱼摆盘 *\n"
            f"/鲸绘取消 用于取消操作\n\n"
            f"默认单次操作，最多连续操作 {self.cfg.max_rounds} 次，未附图时等待 {self.cfg.wait_seconds} 秒\n"
            f"输出为PNG，请保留原图尺寸，避免平台压缩、裁剪或缩放\n"
            f"加混和解混操作都可用来进行混淆，两者互为逆向操作，使用解混进行混淆的图片需要加混才能恢复"
            f"混淆操作可进行多次操作，如果如果原图进行了多次混淆操作，则需要进行同等次数的解混操作"
            f"⚠本功能仅为趣味功能，参考小番茄图片混淆的实现，不具备密码学安全性！仅支持静态图片"
        )

    async def close(self):
        self.closed = True
        self.pending.clear()
        self.prompts.clear()
        for stop in self.stops:
            stop.set()
        tasks = tuple(self.tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self.seen.clear()
