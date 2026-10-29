"""Errors whose messages are safe to show in a chat."""


class UserError(ValueError):
    """Invalid input or a bounded operation rejected by the toolbox."""
