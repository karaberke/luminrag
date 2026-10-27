"""Stored metadata must not turn evidence rendering into a server error."""

import pytest

from backend.utils.timestamps import format_timestamp


@pytest.mark.parametrize("value,expected", [
    (0, "0:00"), (1, "0:01"), (59.9, "0:59"), (60, "1:00"),
    (3661, "61:01"), ("90.8", "1:30"), (" 12.2 ", "0:12"),
])
def test_formats_finite_nonnegative_source_seconds(value, expected):
    assert format_timestamp(value) == expected


@pytest.mark.parametrize("value", [
    None, True, False, "", "not a timestamp", "1:30", [], {},
    float("nan"), float("inf"), float("-inf"), "NaN", "Infinity", "-Infinity",
    -1, -0.5, "-10", "1e10000",
])
def test_invalid_source_metadata_has_no_media_position(value):
    assert format_timestamp(value) is None


def test_oversized_integer_metadata_is_ignored_without_conversion_error():
    assert format_timestamp(10 ** 10000) is None
