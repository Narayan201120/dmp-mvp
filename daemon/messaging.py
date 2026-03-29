from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Any


class MessageBus:
    """Small asyncio message bus used to connect runtime loops."""

    def __init__(self) -> None:
        self._topics: defaultdict[str, asyncio.Queue[Any]] = defaultdict(asyncio.Queue)

    async def publish(self, topic: str, message: Any) -> None:
        await self._topics[topic].put(message)

    async def consume(self, topic: str) -> Any:
        return await self._topics[topic].get()

    def queue(self, topic: str) -> asyncio.Queue[Any]:
        return self._topics[topic]

