"""Linux kernel constants for CFS scheduling (kernel 5.15, single core M=1).

References:
    Yoon et al., "Worst case response time analysis for completely fair
    scheduling in Linux systems", Real-Time Systems, 2025.
"""

from __future__ import annotations

from typing import Final

NICE_0_WEIGHT: Final[int] = 1024
"""Weight corresponding to nice value 0 (w_0 in the paper)."""

NICE_MIN: Final[int] = -20
"""Minimum nice value (highest priority)."""

NICE_MAX: Final[int] = 19
"""Maximum nice value (lowest priority)."""

# Nice-to-weight mapping table from linux/kernel/sched/core.c
# Each step is approximately 1.25x ratio.
NICE_TO_WEIGHT: Final[dict[int, int]] = {
    -20: 88761, -19: 71755, -18: 56483, -17: 46273, -16: 36261,
    -15: 29154, -14: 23254, -13: 18705, -12: 14949, -11: 11916,
    -10: 9548, -9: 7620, -8: 6100, -7: 4904, -6: 3906,
    -5: 3121, -4: 2501, -3: 1991, -2: 1586, -1: 1277,
    0: 1024, 1: 820, 2: 655, 3: 526, 4: 423,
    5: 335, 6: 272, 7: 215, 8: 172, 9: 137,
    10: 110, 11: 87, 12: 70, 13: 56, 14: 45,
    15: 36, 16: 29, 17: 23, 18: 18, 19: 15,
}


def nice_to_weight(nice: int) -> int:
    """Convert nice value [-20, 19] to Linux weight.

    Args:
        nice: Nice value in range [NICE_MIN, NICE_MAX].

    Returns:
        Corresponding weight from the kernel mapping table.

    Raises:
        ValueError: If nice is outside valid range.
    """
    if not NICE_MIN <= nice <= NICE_MAX:
        msg = f"nice must be in [{NICE_MIN}, {NICE_MAX}], got {nice}"
        raise ValueError(msg)
    return NICE_TO_WEIGHT[nice]
