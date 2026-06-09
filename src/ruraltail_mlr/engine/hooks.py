from __future__ import annotations


class Hook:
    def before_epoch(self, *args, **kwargs) -> None:
        return None

    def after_epoch(self, *args, **kwargs) -> None:
        return None
