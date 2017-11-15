#!/usr/bin/env python
# coding=utf-8
import numpy as np
from collections import namedtuple
from typing import Sequence, Tuple

from pygc import great_distance

from ioos_qc.utils import (
    isnan,
    isfixedlength
)


class QartodFlags(object):
    """Primary flags for QARTOD."""

    GOOD = 1
    UNKNOWN = 2
    SUSPECT = 3
    FAIL = 4
    MISSING = 9


N = float  # Union[int, float, Decimal]
span = namedtuple('Span', 'minv maxv')


def qartod_compare(vectors : Sequence[Sequence[N]]
                   ) -> np.ma.MaskedArray:
    """Aggregates an array of glags by precedence into a single array.

    Args:
        vectors: An array of uniform length arrays representing individual flags

    Returns:
        A masked array of aggregated flag data.
    """
    shapes = [v.shape[0] for v in vectors]
    # Assert that all of the vectors are the same size.
    assert all([s == shapes[0] for s in shapes])
    assert all([v.ndim == 1 for v in vectors])

    result = np.ma.empty(shapes[0])
    result.fill(QartodFlags.MISSING)

    priorities = [
        QartodFlags.MISSING,
        QartodFlags.UNKNOWN,
        QartodFlags.GOOD,
        QartodFlags.SUSPECT,
        QartodFlags.FAIL
    ]
    # For each of the priorities in order, set the resultant array to the the
    # flag where that flag exists in each of the vectors.
    for p in priorities:
        for v in vectors:
            idx = np.where(v == p)[0]
            result[idx] = p
    return result


def location_test(lon : Sequence[N],
                  lat : Sequence[N],
                  bbox : Tuple[N, N, N, N] = (-180, -90, 180, 90),
                  range_max : N = None
                  ) -> np.ma.core.MaskedArray:
    """Checks that a location is within reasonable bounds.

    Checks that longitude and latitude are within reasonable bounds defaulting
    to lon = [-180, 180] and lat = [-90, 90].  Optionally, check for a maximum
    range parameter in great circle distance defaulting to meters which can
    also use a unit from the quantities library. Missing and masked data is
    flagged as UNKNOWN.

    Args:
        lon: Longitudes as a numeric numpy array or a list of numbers.
        lat: Latitudes as a numeric numpy array or a list of numbers.
        bbox: A length 4 tuple expressed in (minx, miny, maxx, maxy) [optional].
        range_max: Maximum allowed range expressed in geodesic curve distance (meters).

    Returns:
        A masked array of flag values equal in size to that of the input.
    """

    bboxnt = namedtuple('BBOX', 'minx miny maxx maxy')
    if bbox is not None:
        assert isfixedlength(bbox, 4)
        bbox = bboxnt(*bbox)

    lat = np.ma.masked_invalid(np.array(lat, dtype=np.float64))
    lon = np.ma.masked_invalid(np.array(lon, dtype=np.float64))

    if lon.shape != lat.shape:
        raise ValueError(
            'Lon ({0.shape}) and lat ({1.shape}) are different shapes'.format(
                lon, lat
            )
        )

    # Save original shape
    original_shape = lon.shape
    lon = lon.flatten()
    lat = lat.flatten()

    # Start with everything as passing (1)
    flag_arr = np.ma.ones(lon.size, dtype='uint8')

    # If either lon or lat are masked we just set the flag to MISSING
    mloc = lon.mask & lat.mask
    flag_arr[mloc] = QartodFlags.MISSING

    # If there is only one masked value fail the location test
    mismatch = lon.mask != lat.mask
    flag_arr[mismatch] = QartodFlags.FAIL

    if range_max is not None and lon.size > 1:
        # Calculating the great_distance between each point
        # Flag suspect any distance over range_max
        d = np.ma.zeros(lon.size, dtype=np.float64)
        d[1:] = great_distance(
            start_latitude=lat[:-1],
            end_latitude=lat[1:],
            start_longitude=lon[:-1],
            end_longitude=lon[1:]
        )['distance']
        flag_arr[d > range_max] = QartodFlags.SUSPECT

    # Ignore warnings when comparing NaN values even though they are masked
    # https://github.com/numpy/numpy/blob/master/doc/release/1.8.0-notes.rst#runtime-warnings-when-comparing-nan-numbers
    with np.errstate(invalid='ignore'):
        flag_arr[(lon < bbox.minx) | (lat < bbox.miny) |
                 (lon > bbox.maxx) | (lat > bbox.maxy)] = QartodFlags.FAIL

    return flag_arr.reshape(original_shape)


def gross_range_test(inp : Sequence[N],
                     fail_span : Tuple[N, N],
                     suspect_span : Tuple[N, N] = None
                     ) -> np.ma.core.MaskedArray:
    """Checks that values are within reasonable range bounds.

    Given a 2-tuple of minimum/maximum values, flag data outside of the given
    range as FAIL data.  Optionally also flag data which falls outside of a user
    defined range as SUSPECT. Missing and masked data is flagged as UNKNOWN.

    Args:
        inp: Input data as a numeric numpy array or a list of numbers.
        fail_span: 2-tuple range which to flag outside data as FAIL.
        suspect_span: 2-tuple range which to flag outside data as SUSPECT. [optional]

    Returns:
        A masked array of flag values equal in size to that of the input.
    """

    assert isfixedlength(fail_span, 2)
    sspan = span(*sorted(fail_span))

    inp = np.ma.masked_invalid(np.array(inp, dtype=np.float64))
    # Save original shape
    original_shape = inp.shape
    inp = inp.flatten()
    # Start with everything as passing (1)
    flag_arr = np.ma.ones(inp.size, dtype='uint8')

    # If the value is masked set the flag to MISSING
    flag_arr[inp.mask] = QartodFlags.MISSING

    if suspect_span is not None:
        assert isfixedlength(suspect_span, 2)
        uspan = span(*sorted(suspect_span))
        if uspan.minv < sspan.minv or uspan.maxv > sspan.maxv:
            raise ValueError('User span range may not exceed sensor span')
        # Flag suspect outside of user span
        with np.errstate(invalid='ignore'):
            flag_arr[(inp < uspan.minv) | (inp > uspan.maxv)] = QartodFlags.SUSPECT

    # Flag suspect outside of sensor span
    with np.errstate(invalid='ignore'):
        flag_arr[(inp < sspan.minv) | (inp > sspan.maxv)] = QartodFlags.FAIL

    return flag_arr.reshape(original_shape)


class ClimatologyConfig(object):

    mem = namedtuple('window', [
        'tspan',
        'vspan',
        'zspan'
    ])

    def __init__(self, members=None):
        members = members or []
        self._members = members

    @property
    def members(self):
        return self._members

    def values(self, tind, zind=None):
        span = (None, None)
        for m in self._members:
            # If we are between times
            if tind > m.tspan.minv and tind <= m.tspan.maxv:
                if not isnan(zind) and not isnan(m.zspan):
                    # If we are between depths
                    if zind > m.zspan.minv and zind <= m.zspan.maxv:
                        span = m.vspan
                elif isnan(zind) and isnan(m.zspan):
                    span = m.vspan
        return span

    def add(self,
            tspan : Tuple[N, N],
            vspan : Tuple[N, N],
            zspan : Tuple[N, N] = None) -> None:

        assert isfixedlength(tspan, 2)
        tspan = span(*sorted(tspan))

        assert isfixedlength(vspan, 2)
        vspan = span(*sorted(vspan))

        if zspan is not None:
            assert isfixedlength(zspan, 2)
            zspan = span(*sorted(zspan))

        self._members.append(
            self.mem(
                tspan,
                vspan,
                zspan
            )
        )


def climatology_test(config : ClimatologyConfig,
                     tinp : Sequence[N],
                     vinp : Sequence[N],
                     zinp : Sequence[N],
                     ) -> np.ma.core.MaskedArray:
    """Checks that values are within reasonable range bounds and flags as SUSPECT.

    Data for which no ClimatologyConfig member exists is marked as UNKNOWN.

    Args:
        config: A ClimatologyConfig object.
        tinp: Time data as a numpy array of dtype `datetime64`.
        vinp: Input data as a numeric numpy array or a list of numbers.
        zinp: Z (depth) data as a numeric numpy array or a list of numbers.

    Returns:
        A masked array of flag values equal in size to that of the input.
    """

    tinp = np.array(tinp)
    vinp = np.ma.masked_invalid(np.array(vinp, dtype=np.float64))
    zinp = np.ma.masked_invalid(np.array(zinp, dtype=np.float64))

    # Save original shape
    original_shape = vinp.shape

    tinp = tinp.flatten()
    vinp = vinp.flatten()
    zinp = zinp.flatten()

    # Start with everything as passing (1)
    flag_arr = np.ma.ones(vinp.size, dtype='uint8')

    # If the value is masked set the flag to MISSING
    flag_arr[vinp.mask] = QartodFlags.MISSING

    for i, (tind, vind, zind) in enumerate(zip(tinp, vinp, zinp)):
        minv, maxv = config.values(tind, zind)
        if minv is None or maxv is None:
            flag_arr[i] = QartodFlags.MISSING
        else:
            # Flag suspect outside of climatology span
            with np.errstate(invalid='ignore'):
                if vind < minv or vind > maxv:
                    flag_arr[i] = QartodFlags.SUSPECT

    return flag_arr.reshape(original_shape)


def spike_test(inp : Sequence[N],
               thresholds : Tuple[N, N]
               ) -> np.ma.core.MaskedArray:
    """Check for spikes by checking neigboring data against thresholds

    Determine if there is a spike at data point n-1 by subtracting
    the midpoint of n and n-2 and taking the absolute value of this
    quantity, and checking if it exceeds a a low or high threshold.
    Values which do not exceed either threshold are flagged GOOD,
    values which exceed the low threshold are flagged SUSPECT,
    and values which exceed the high threshold are flagged FAIL.
    Missing and masked data is flagged as UNKNOWN.

    Args:
        inp: Input data as a numeric numpy array or a list of numbers.
        thresholds: 2-tuple threshold range.  The lower number will always
            represent the SUSPECT threshold value and the higher number will
            always represent the FAIL threshold value.

    Returns:
        A masked array of flag values equal in size to that of the input.
    """

    assert isfixedlength(thresholds, 2)
    thresholds = span(*sorted([ abs(x) for x in thresholds] ))

    inp = np.ma.masked_invalid(np.array(inp, dtype=np.float64))
    # Save original shape
    original_shape = inp.shape
    inp = inp.flatten()

    # Calculate the average of n-2 and n
    ref = np.zeros(inp.size, dtype=np.float64)
    ref[1:-1] = np.abs((inp[0:-2] + inp[2:]) / 2)
    ref = np.ma.masked_invalid(ref)

    # Start with everything as passing (1)
    flag_arr = np.ma.ones(inp.size, dtype='uint8')

    # Calculate the (n-1 - ref) difference
    diff = np.abs(inp - ref)

    # If n-1 - ref is greater than the low threshold, SUSPECT test
    with np.errstate(invalid='ignore'):
        flag_arr[diff > thresholds.minv] = QartodFlags.SUSPECT

    # If n-1 - ref is greater than the high threshold, FAIL test
    with np.errstate(invalid='ignore'):
        flag_arr[diff > thresholds.maxv] = QartodFlags.FAIL

    # If the value is masked or nan set the flag to MISSING
    flag_arr[diff.mask] = QartodFlags.MISSING

    return flag_arr.reshape(original_shape)


def rate_of_change_test(inp : Sequence[N],
                        deviation : N,
                        num_deviations : int = 3
                        ) -> np.ma.core.MaskedArray:
    """Checks the difference of neighboring values against N number of standard deviations.

    Checks the first order difference of a series of values to see if
    there are any values exceeding a standard deviation threshold defined
    by the inputs.  These are then marked as SUSPECT.  It is up to the test operator
    to determine an appropriate threshold value for the absolute difference not to
    exceed. Threshold are expressed as a standard deviation threshold and a factor
    controlling how many of the thresholds to tolerate before flagging.
    Missing and masked data is flagged as UNKNOWN.

    Args:
        inp: Input data as a numeric numpy array or a list of numbers.
        deviation: The number to multiply by `num_deviations` to get the threshold value.
        num_deviations: The number to multiple by `deviation` to get the threshold value.
            Defaults to 3.

    Returns:
        A masked array of flag values equal in size to that of the input.
    """

    inp = np.ma.masked_invalid(np.array(inp, dtype=np.float64))
    # Save original shape
    original_shape = inp.shape
    inp = inp.flatten()

    # Start with everything as passing (1)
    flag_arr = np.ma.ones(inp.size, dtype='uint8')

    # Calculate the (n - n-1) difference.
    diff = np.abs(inp[1:] - inp[:-1])
    # First value set to zero (can't subtract previous element)
    diff = np.append(
        np.zeros(1, dtype=np.float64),
        diff
    )

    final_threshold = deviation * num_deviations
    # If n - n-1 is greater than the set deviations, SUSPECT test
    with np.errstate(invalid='ignore'):
        flag_arr[diff > final_threshold] = QartodFlags.SUSPECT

    # If the value is masked set the flag to MISSING
    flag_arr[diff.mask] = QartodFlags.MISSING

    return flag_arr.reshape(original_shape)


def flat_line_test(inp : Sequence[N],
                   counts : Tuple[int, int],
                   tolerance : N = 0
                   ) -> np.ma.MaskedArray:
    """Check for consecutively repeated values within a tolerance.

    Missing and masked data is flagged as UNKNOWN.

    Args:
        inp: Input data as a numeric numpy array or a list of numbers.
        counts: 2-tuple representing the number of repetitions within `tolerance` to
            allow before being flagged as SUSPECT or FAIL. The larger of the two
            values is alwasy the SUSPECT threshold and the lower of the two numbers
            is always the FAIL threshold.
        tolerance: The tolerance that should be exceeded between consecutive values.
            If the number consecutive values occuring that don't cross over `tolerance`
            cross over either of the `counts` then the data will be flagged.

    Returns:
        A masked array of flag values equal in size to that of the input.
    """

    def chunk(a, num):
        out = np.ma.masked_all(
            (a.size, num),
            dtype=np.float64
        )
        for i in reversed(range(0, a.size)):
            start = max(0, i - num)
            data = a[start:i]
            out[i, :data.size] = data
        return out

    assert isfixedlength(counts, 2)
    counts = span(*sorted(counts))
    if not isinstance(counts.minv, int) or not isinstance(counts.maxv, int):
        raise TypeError('Counts must be integers. Got {}'.format(counts))

    inp = np.ma.masked_invalid(np.array(inp, dtype=np.float64))
    # Save original shape
    original_shape = inp.shape
    inp = inp.flatten()

    # Start with everything as passing (1)
    flag_arr = np.ma.ones(inp.size, dtype='uint8')

    suspect_chunks = chunk(inp, counts.minv)
    suspect_data = np.repeat(inp, counts.minv).reshape(inp.size, counts.minv)
    with np.errstate(invalid='ignore'):
        suspect_test = np.all(np.abs((suspect_chunks - suspect_data)) < tolerance, axis=1)
        suspect_test = np.ma.filled(suspect_test, fill_value=False)
        flag_arr[suspect_test] = QartodFlags.SUSPECT

    fail_chunks = chunk(inp, counts.maxv)
    failed_data = np.repeat(inp, counts.maxv).reshape(inp.size, counts.maxv)
    with np.errstate(invalid='ignore'):
        failed_test = np.all(np.abs((fail_chunks - failed_data)) < tolerance, axis=1)
        failed_test = np.ma.filled(failed_test, fill_value=False)
        flag_arr[failed_test] = QartodFlags.FAIL

    # If the value is masked set the flag to MISSING
    flag_arr[inp.mask] = QartodFlags.MISSING

    return flag_arr.reshape(original_shape)


def attenuated_signal_test(inp : Sequence[N],
                           threshold : Tuple[N, N],
                           check_type : str = 'std'
                           ) -> np.ma.MaskedArray:
    """Check for near-flat-line conditions using a range or standard deviation.

    Missing and masked data is flagged as UNKNOWN.

    Args:
        inp: Input data as a numeric numpy array or a list of numbers.
        threshold: 2-tuple representing the minimum thresholds to use for SUSPECT
            and FAIL checks. The smaller of the two values is used in the SUSPECT
            tests and the greater of the two values is used in the FAIL tests.
        check_type: Either 'std' (default) or 'range', depending on the type of check
            you wish to perform.

    Returns:
        A masked array of flag values equal in size to that of the input.
        This array will always contain only a single unique value since all
        input data is flagged together.
    """

    assert isfixedlength(threshold, 2)
    threshold = span(*reversed(sorted(threshold)))

    inp = np.ma.masked_invalid(np.array(inp, dtype=np.float64))
    # Save original shape
    original_shape = inp.shape
    inp = inp.flatten()

    if check_type == 'std':
        check_val = np.std(inp)
    elif check_type == 'range':
        check_val = np.ptp(inp)
    else:
        raise ValueError('Check type "{}" is not defined'.format(check_type))

    # Start with everything as passing (1)
    flag_arr = np.ma.ones(inp.size, dtype='uint8')

    if check_val < threshold.maxv:
        flag_arr.fill(QartodFlags.FAIL)
    elif check_val < threshold.minv:
        flag_arr.fill(QartodFlags.SUSPECT)

    # If the value is masked set the flag to MISSING
    flag_arr[inp.mask] = QartodFlags.MISSING

    return flag_arr.reshape(original_shape)