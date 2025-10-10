"""Tests for statistics functions within the Model layer."""

import pytest
import numpy as np
import numpy.testing as npt

from inflammation.models import daily_mean, daily_max, daily_min, patient_normalise

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

@pytest.mark.parametrize(
    "test, expected, expect_raises",
[
        ([[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]], None),
        ([[1, 1, 1], [1, 1, 1], [1, 1, 1]], [[1, 1, 1], [1, 1, 1], [1, 1, 1]], None),
        ([[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[0.33, 0.67, 1], [0.67, 0.83, 1], [0.78, 0.89, 1]], None),
        ([[-1, 2, 3], [4, 5, 6], [7, 8, 9]], None, ValueError),

    ])
def test_patient_normalise(test, expected, expect_raises):
    """Test normalisation works for arrays of one and positive integers.
       Test with a relative and absolute tolerance of 0.01."""

    if isinstance(test, list):
        test = np.array(test)
    if expect_raises is not None:
        with pytest.raises(expect_raises):
            patient_normalise(test)
    else:
        result = patient_normalise(test)
        npt.assert_allclose(result, np.array(expected), rtol=1e-2, atol=1e-2)