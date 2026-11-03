"""Process-wide mutable state: the loaded model bundle and the epoch sentinel."""

from datetime import datetime, timezone


# ── Shared state ──────────────────────────────────────────────────────────────
state: dict = {}


EPOCH = datetime.min.replace(tzinfo=timezone.utc)
