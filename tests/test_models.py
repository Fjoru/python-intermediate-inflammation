"""Tests for statistics functions within the Model layer."""

import pytest
import numpy as np
import numpy.testing as npt

from inflammation.models import daily_mean, daily_max, daily_min

@pytest.mark.parametrize(
    "test_input,expected",
    [
        (np.array([[0, 0], [0, 0], [0, 0]]), np.array([0, 0])),
        (np.array([[1, 2], [3, 4], [5, 6]]), np.array([3, 4]))
    ]
)
def test_daily_mean(test_input, expected):
    """Test that mean function works for arrays of zeros and positive integers."""
    npt.assert_array_equal(daily_mean(test_input), expected)

@pytest.mark.parametrize(
    "test_input,expected",
    [
        (np.array([[0, 0], [0, 0], [0, 0]]), np.array([0, 0])),
        (np.array([[1, 2], [3, 4], [5, 6]]), np.array([5, 6]))
        
    ]
)
def test_daily_max(test_input, expected):
    """Test that max function works for arrays of zeros and positive integers."""
    npt.assert_array_equal(daily_max(test_input), expected)

@pytest.mark.parametrize(
    "test_input,expected",
    [
        (np.array([[0, 0], [0, 0], [0, 0]]), np.array([0, 0])),
        (np.array([[1, 2], [3, 4], [5, 6]]), np.array([1, 2]))
    ]
)

def test_daily_min(test_input, expected):
    """Test that min function works for arrays of zeros and positive integers."""
    npt.assert_array_equal(daily_min(test_input), expected)


def test_daily_min_string():
    """Test for TypeError when passing strings"""

    with pytest.raises(TypeError):
        error_expected = daily_min([['Hello', 'there'], ['General', 'Kenobi']])
