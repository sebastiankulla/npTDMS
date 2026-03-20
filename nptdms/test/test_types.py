"""Test type reading and writing"""

from datetime import date, datetime
import io
import numpy as np
import pytest

from nptdms import types
from nptdms.types import StructType


@pytest.mark.parametrize(
    "time_string",
    [
        pytest.param('2019-11-08T18:47:00', id="standard timestamp"),
        pytest.param('0000-01-01T05:00:00', id="timestamp before TDMS epoch"),
        pytest.param('2019-11-08T18:47:00.123456', id="timestamp with microseconds"),
        pytest.param('1903-12-31T23:59:59.500', id="timestamp before TDMS epoch with microseconds"),
    ]
)
def test_timestamp_round_trip(time_string):
    expected_datetime = np.datetime64(time_string)

    timestamp = types.TimeStamp(expected_datetime)
    data_file = io.BytesIO(timestamp.bytes)

    read_datetime = types.TimeStamp.read(data_file).as_datetime64()

    assert expected_datetime == read_datetime


def test_timestamp_from_datetime():
    """Test timestamp from built in datetime value"""

    input_datetime = datetime(2019, 11, 8, 18, 47, 0)
    expected_datetime = np.datetime64('2019-11-08T18:47:00')

    timestamp = types.TimeStamp(input_datetime)
    data_file = io.BytesIO(timestamp.bytes)

    read_datetime = types.TimeStamp.read(data_file)

    assert expected_datetime == read_datetime.as_datetime64()


def test_timestamp_from_date():
    """Test timestamp from built in date value"""

    input_datetime = date(2019, 11, 8)
    expected_datetime = np.datetime64('2019-11-08T00:00:00')

    timestamp = types.TimeStamp(input_datetime)
    data_file = io.BytesIO(timestamp.bytes)

    read_datetime = types.TimeStamp.read(data_file)

    assert expected_datetime == read_datetime.as_datetime64()


# Tests for numpy 2.0 compatibility: StructType.from_bytes() must not use
# the deprecated `array.dtype = ...` assignment (removed in numpy 2.0).

@pytest.mark.parametrize("tdms_type,values", [
    pytest.param(types.Int8,   [-128, -1, 0, 1, 127],              id="Int8"),
    pytest.param(types.Int16,  [-32768, -1, 0, 1, 32767],          id="Int16"),
    pytest.param(types.Int32,  [-(2**31), -1, 0, 1, 2**31 - 1],   id="Int32"),
    pytest.param(types.Int64,  [-(2**63), -1, 0, 1, 2**63 - 1],   id="Int64"),
    pytest.param(types.Uint8,  [0, 1, 127, 255],                   id="Uint8"),
    pytest.param(types.Uint16, [0, 1, 32767, 65535],               id="Uint16"),
    pytest.param(types.Uint32, [0, 1, 2**31 - 1, 2**32 - 1],      id="Uint32"),
    pytest.param(types.Uint64, [0, 1, 2**63 - 1, 2**64 - 1],      id="Uint64"),
    pytest.param(types.SingleFloat, [-1.0, 0.0, 1.0],              id="SingleFloat"),
    pytest.param(types.DoubleFloat, [-1.0, 0.0, 1.0],              id="DoubleFloat"),
])
def test_from_bytes_little_endian(tdms_type, values):
    """from_bytes produces correct values for little-endian data (numpy 2.0)"""
    expected = np.array(values, dtype=tdms_type.nptype)
    raw = np.frombuffer(expected.tobytes(), dtype=np.uint8)
    result = tdms_type.from_bytes(raw, "<")
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("tdms_type,values", [
    pytest.param(types.Int8,   [-128, -1, 0, 1, 127],           id="Int8"),
    pytest.param(types.Int16,  [-32768, -1, 0, 1, 32767],       id="Int16"),
    pytest.param(types.Int32,  [-(2**31), -1, 0, 1, 2**31 - 1], id="Int32"),
    pytest.param(types.Uint8,  [0, 1, 127, 255],                id="Uint8"),
    pytest.param(types.Uint16, [0, 1, 32767, 65535],            id="Uint16"),
    pytest.param(types.Uint32, [0, 1, 2**31 - 1, 2**32 - 1],   id="Uint32"),
    pytest.param(types.SingleFloat, [-1.0, 0.0, 1.0],           id="SingleFloat"),
    pytest.param(types.DoubleFloat, [-1.0, 0.0, 1.0],           id="DoubleFloat"),
])
def test_from_bytes_big_endian(tdms_type, values):
    """from_bytes produces correct values for big-endian data (numpy 2.0)"""
    expected = np.array(values, dtype=tdms_type.nptype)
    raw = np.frombuffer(expected.byteswap().tobytes(), dtype=np.uint8)
    result = tdms_type.from_bytes(raw, ">")
    np.testing.assert_array_equal(result, expected)


def test_from_bytes_int8_boundary_values():
    """Int8.from_bytes handles boundary values correctly — original bug triggered on int8"""
    values = np.array([-128, -1, 0, 1, 127], dtype=np.int8)
    raw = np.frombuffer(values.tobytes(), dtype=np.uint8)
    result = types.Int8.from_bytes(raw, "<")
    assert result.dtype == np.dtype("int8")
    np.testing.assert_array_equal(result, values)


def test_from_bytes_uint32_large_values():
    """Uint32.from_bytes handles values > 2^31 without overflow"""
    values = np.array([2**31, 2**32 - 1], dtype=np.uint32)
    raw = np.frombuffer(values.tobytes(), dtype=np.uint8)
    result = types.Uint32.from_bytes(raw, "<")
    assert result.dtype == np.dtype("uint32")
    np.testing.assert_array_equal(result, values)
