"""Pending tickets: processed tickets held in memory while the requester tries
the AI's troubleshooting steps.

A held ticket is not in tickets.json and not on the dashboard. It is saved only
if the requester answers "still not fixed"; if they answer "fixed", it is
dropped. The ticket is kept here (not sent back through the browser) so the
requester cannot change its priority or department, and so the AI is not asked
to classify the same ticket twice.

In memory only: held tickets are lost if the server restarts, and expire after
`ttl_seconds`. A requester whose ticket is gone is asked to submit again.
"""

import secrets
import threading
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Callable
from typing import Any

DEFAULT_TTL_SECONDS = 3600
DEFAULT_MAX_HELD = 500


class PendingTicketStore(ABC):
    """The contract for holding tickets while the requester tries the steps.
    io_manager depends only on this, so the storage can be swapped (for
    example for one that survives a restart) without changing it."""

    @abstractmethod
    def hold(self, ticket: Any) -> str:
        """Keeps the ticket and returns its token."""

    @abstractmethod
    def take(self, token: str) -> Any | None:
        """Removes and returns the ticket, or None if unknown or expired."""

    @abstractmethod
    def put_back(self, token: str, ticket: Any) -> None:
        """Returns a taken ticket, so the requester can try again."""


class InMemoryPendingTickets(PendingTicketStore):
    """Thread-safe in-memory store, each ticket found by a random token."""

    def __init__(self, ttl_seconds: float = DEFAULT_TTL_SECONDS,
                 max_held: int = DEFAULT_MAX_HELD,
                 clock: Callable[[], float] = time.time) -> None:
        self._ttl_seconds = ttl_seconds
        self._max_held = max_held
        self._clock = clock
        self._held: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._lock = threading.Lock()

    def hold(self, ticket: Any) -> str:
        token = secrets.token_urlsafe(16)
        self._put(token, ticket)
        return token

    def take(self, token: str) -> Any | None:
        # Taking removes it, so two answers cannot save the ticket twice.
        with self._lock:
            self._drop_expired()
            entry = self._held.pop(token, None)
        return entry[1] if entry else None

    def put_back(self, token: str, ticket: Any) -> None:
        self._put(token, ticket)

    def __len__(self) -> int:
        with self._lock:
            self._drop_expired()
            return len(self._held)

    def _put(self, token: str, ticket: Any) -> None:
        with self._lock:
            self._drop_expired()
            self._held[token] = (self._clock(), ticket)
            while len(self._held) > self._max_held:
                self._held.popitem(last=False)  # the oldest goes first

    def _drop_expired(self) -> None:
        """Caller holds the lock. Entries are in the order they were added,
        so only the oldest ones can have expired."""
        now = self._clock()
        while self._held:
            token, (held_at, _) = next(iter(self._held.items()))
            if now - held_at < self._ttl_seconds:
                break
            del self._held[token]
