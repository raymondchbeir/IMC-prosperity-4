from __future__ import annotations

from app.rounds import round1

ROUND_PLUGIN_REGISTRY = {
    1: round1,
}


def get_round_plugin(round_number: int | None):
    if round_number is None:
        return None
    return ROUND_PLUGIN_REGISTRY.get(int(round_number))