from __future__ import annotations

import contextvars

from openai import AsyncOpenAI

# A caller may pin all requests of one task to one replica (set this to the task id): the task's shared prefix
# (system prompt + reference image) then stays in a single engine's prefix cache. None = plain round-robin.
affinity_key: contextvars.ContextVar[str | None] = contextvars.ContextVar("llm_affinity_key", default=None)


class LoadBalancedClient:
    """Weighted round-robin over AsyncOpenAI clients serving the same model.
    """

    def __init__(self, clients: list[AsyncOpenAI], weights: list[int]) -> None:
        if len(clients) != len(weights):
            raise ValueError("clients and weights must have the same length")
        schedule: list[AsyncOpenAI] = []
        # Interleave by weight (e.g. weights [2,1] -> [a, b, a]) so bursts
        # don't land on a single replica.
        counters = [0] * len(clients)
        total = sum(max(1, w) for w in weights)
        for step in range(total):
            best = max(
                range(len(clients)),
                key=lambda i: max(1, weights[i]) / (counters[i] + 1),
            )
            counters[best] += 1
            schedule.append(clients[best])
        self._schedule = schedule
        self._i = 0
        self._clients = list(clients)
        self._load = [0] * len(clients)          # tasks pinned to each replica
        self._pinned: dict[str, int] = {}

    @property
    def chat(self):
        key = affinity_key.get()
        if key is not None:
            idx = self._pinned.get(key)
            if idx is None:                       # a new task goes to the replica with the fewest pinned tasks
                idx = min(range(len(self._clients)), key=lambda i: (self._load[i], i))
                self._pinned[key] = idx
                self._load[idx] += 1
            return self._clients[idx].chat
        client = self._schedule[self._i % len(self._schedule)]
        self._i += 1
        return client.chat
