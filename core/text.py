"""Four-symbol UTF-16 beast encoding with self-describing alphabet framing."""

import re
import unicodedata

from .errors import UserError

DEFAULT_ALPHABET = "嗡呜隆嗷"


def validate_alphabet(value: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 4
        or len(set(value)) != 4
        or any(c.isspace() or unicodedata.category(c).startswith("C") for c in value)
    ):
        raise UserError("加密词必须是四个互不重复的可见字符，不能包含空白。")
    return value


def encode(text: str, alphabet: str = DEFAULT_ALPHABET) -> str:
    validate_alphabet(alphabet)
    if not text:
        raise UserError("请输入需要加密的文字。")
    try:
        raw = text.encode("utf-16-be").hex()
    except UnicodeError as exc:
        raise UserError("文字包含无效 Unicode 字符。") from exc
    body = []
    for i, digit in enumerate(raw):
        value = (int(digit, 16) + i) % 16
        body.append(alphabet[value // 4] + alphabet[value % 4])
    return alphabet[3] + alphabet[1] + alphabet[0] + "".join(body) + alphabet[2]


def decode(text: str, alphabet: str | None = None) -> str:
    if len(text) < 12 or (len(text) - 4) % 8:
        raise UserError("密文长度不正确，请保留完整首尾标记。")
    inferred = text[2] + text[1] + text[-1] + text[0]
    validate_alphabet(inferred)
    if alphabet is not None and validate_alphabet(alphabet) != inferred:
        raise UserError("指定加密词与密文首尾标记不一致。")
    digits = {c: i for i, c in enumerate(inferred)}
    body = text[3:-1]
    if set(body) - digits.keys():
        raise UserError("密文包含字表之外的字符，请勿插入空格或换行。")
    hex_digits = []
    for i in range(0, len(body), 2):
        value = (digits[body[i]] * 4 + digits[body[i + 1]] - i // 2) % 16
        hex_digits.append(format(value, "x"))
    try:
        return bytes.fromhex("".join(hex_digits)).decode("utf-16-be")
    except UnicodeError as exc:
        raise UserError("密文无法还原为有效文字，可能已经损坏。") from exc


def parse_text(value: str, presets: dict[str, str]) -> tuple[str | None, str]:
    """Return an explicit alphabet, or None for default/auto mode."""
    if not value:
        raise UserError(
            "请提供正文，例如 /鲸言加密 你好；含空格正文用 /鲸言加密 -- hello world。"
        )
    first, *rest = re.split(r"\s+", value, maxsplit=1)
    if first == "--":
        if not rest or not rest[0]:
            raise UserError("-- 后面需要正文。")
        return None, rest[0]
    if not rest:
        return None, first
    if len(first) == 2:
        if first not in presets:
            raise UserError(f"未知预设“{first}”；默认字表的含空格正文请放在 -- 后。")
        return presets[first], rest[0]
    return validate_alphabet(first), rest[0]
