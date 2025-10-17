import os
import numpy as np
import numpy.testing as npt
import math
import pytest
from unittest.mock import Mock
from inflammation.compute_data import analyse_data, compute_standard_deviation_by_day, CSVDataSource


def test_analyse_data_mock_source():
    data_source = Mock()

    data_source.load_inflammation_data.return_value = [
        np.array([[0, 1, 2], [3, 4, 5]]),
        np.array([[6, 7, 8], [9, 10, 11]])
    ]

    analyse_data(data_source)


@pytest.mark.parametrize('data,expected_output', [
    ([[[0, 1, 0], [0, 2, 0]]], [0, 0, 0]),
    ([[[0, 2, 0]], [[0, 1, 0]]], [0, math.sqrt(0.25), 0]),
    ([[[0, 1, 0], [0, 2, 0]], [[0, 1, 0], [0, 2, 0]]], [0, 0, 0])
],
ids=['Two patients in same file', 'Two patients in different files', 'Two identical patients in two different files'])
def test_compute_standard_deviation_by_day(data, expected_output):

    result = compute_standard_deviation_by_day(data)
    npt.assert_array_almost_equal(result, expected_output)


def test_analyse_data():
    path = os.path.join( os.getcwd(), "../data")
    data_source = CSVDataSource(path)
    result = analyse_data(data_source)

    # TODO: add assert statement(s) to test the result value is as expected