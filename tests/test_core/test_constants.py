"""Tests for core constants module.

Tests cover:
    - NICE_0_WEIGHT constant
    - NICE_TO_WEIGHT mapping table
    - nice_to_weight() conversion
    - Nice value boundary conditions
"""

from __future__ import annotations

import pytest

from cfs_wcrt import NICE_0_WEIGHT, NICE_MAX, NICE_MIN, NICE_TO_WEIGHT, nice_to_weight


class TestConstants:
    """Sanity checks on module-level constants."""

    def test_nice_0_weight(self) -> None:
        assert NICE_0_WEIGHT == 1024

    def test_nice_bounds(self) -> None:
        assert NICE_MIN == -20
        assert NICE_MAX == 19

    def test_to_weight_table_length(self) -> None:
        """Must have exactly 40 entries (-20 through 19)."""
        assert len(NICE_TO_WEIGHT) == 40

    def test_to_weight_table_monotonic(self) -> None:
        """Weights must be strictly decreasing as nice increases."""
        weights = [NICE_TO_WEIGHT[n] for n in range(NICE_MIN, NICE_MAX + 1)]
        for i in range(len(weights) - 1):
            assert weights[i] >= weights[i + 1], f"Not monotonic at index {i}"
        # At least some must be strictly decreasing
        assert weights[0] > weights[-1]

    def test_to_weight_specific_values(self) -> None:
        """Spot-check known kernel values."""
        assert NICE_TO_WEIGHT[-20] == 88761  # highest priority
        assert NICE_TO_WEIGHT[0] == 1024  # default
        assert NICE_TO_WEIGHT[19] == 15  # lowest priority


class TestNiceToWeight:
    """Conversion function behaviour."""

    def test_convert_zero(self) -> None:
        assert nice_to_weight(0) == 1024

    def test_convert_negative(self) -> None:
        assert nice_to_weight(-20) == 88761
        assert nice_to_weight(-5) == 3121

    def test_convert_positive(self) -> None:
        assert nice_to_weight(10) == 110
        assert nice_to_weight(19) == 15

    def test_invalid_below_min(self) -> None:
        with pytest.raises(ValueError, match="must be in"):
            nice_to_weight(-21)

    def test_invalid_above_max(self) -> None:
        with pytest.raises(ValueError, match="must be in"):
            nice_to_weight(20)

    def test_boundary_min(self) -> None:
        assert nice_to_weight(-20) == 88761

    def test_boundary_max(self) -> None:
        assert nice_to_weight(19) == 15
