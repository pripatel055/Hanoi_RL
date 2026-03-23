"""Random-action baseline policy."""

from __future__ import annotations

import random
from collections.abc import Sequence


class RandomPolicy:
    """Sample uniformly from either all actions or a provided valid-action subset."""

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def predict(self, valid_actions: Sequence[int] | None = None) -> int:
        """Return a sampled action index."""
        if valid_actions:
            return int(self._rng.choice(list(valid_actions)))
        return self._rng.randrange(6)
