#!/usr/bin/env python
# coding=utf-8
"""Tests based on the IOOS QARTOD manuals."""
import logging
import warnings
from typing import Dict, List, Tuple, Union, Sequence
from numbers import Real as N
from collections import namedtuple

import numpy as np
import pandas as pd

from ioos_qc.utils import (
    isnan,
    isfixedlength,
    add_flag_metadata,
    great_circle_distance,
    mapdates
)
from ioos_qc.qartod import QartodFlags

L = logging.getLogger(__name__)  # noqa


FLAGS = QartodFlags  # Default name for all check modules
NOTEVAL_VALUE = QartodFlags.UNKNOWN

span = namedtuple('Span', 'minv maxv')



@add_flag_metadata(standard_name='gross_range_test_quality_flag',
                   long_name='Gross Range Test Quality Flag')
def valid_range_test(inp : Sequence[any],
                     valid_span : Tuple[any, any],
                     dtype : np.dtype = None,
                     start_inclusive : bool = True,
                     end_inclusive : bool = False,
                     ) -> np.ma.core.MaskedArray:
    """Checks that values are within a min/max range. This is not unlike a `qartod.gross_range_test`
    with fail and suspect bounds being equal, except that here we specify the inclusive range that
    should pass instead of the exclusive bounds which should fail. This also supports datetime-like
    objects where as the `qartod.gross_range_test` method only supports numerics.

    Given a 2-tuple of minimum/maximum values, flag data outside of the given
    range as FAIL data. Missing and masked data is flagged as UNKNOWN. The first span value is
    treated as inclusive and the second span valid is treated as exclusive. To change this
    behavior you can use the paramters `start_inclusive` and `end_inclusive`.

    Args:
        inp (Sequence[any]): Data as a sequence of objects compatible with the fail_span objects
        fail_span (Tuple[any, any]): 2-tuple range which to flag outside data as FAIL. Objects
            should be of equal format to that of the inp parameter as they will be checked for
            equality without type conversion.
        dtype (np.dtype): Optional. If your data is not already numpy-typed you can specify its
            dtype here.
        start_inclusive (bool): Optional. If the starting span value should be inclusive (True) or
            exclusive (False).
        end_inclusive (bool): Optional. If the ending span value should be inclusive (True) or
            exclusive (False).

    Returns:
        np.ma.core.MaskedArray: A masked array of flag values equal in size to that of the input.
    """

    if dtype is None and hasattr(inp, 'dtype'):
        dtype = inp.dtype

    if dtype is None:
        L.warning("Trying to guess data input type, try specifying the dtype parameter")
        # Try to figure out the dtype so masked values can be calculated
        try:
            # Try datetime-like objects
            inp = np.ma.masked_invalid(mapdates(inp))
            valid_span = np.ma.masked_invalid(mapdates(valid_span))
        except BaseException:
            try:
                # Try floating point numbers
                inp = np.ma.masked_invalid(np.array(inp).astype(np.floating))
                valid_span = np.ma.masked_invalid(np.array(valid_span).astype(np.floating))
            except BaseException:
                # Well, we tried.
                raise ValueError(
                    "Could not determine the type of input, try using the dtype parameter")
    else:
        inp = np.ma.masked_invalid(np.array(inp, dtype=dtype))
        valid_span =  np.ma.masked_invalid(np.array(valid_span, dtype=dtype))

    # Save original shape
    original_shape = inp.shape
    inp = inp.flatten()

    # Start with everything as passing (1)
    flag_arr = np.ma.ones(inp.size, dtype='uint8')

    # Set fail on either side of the bounds, inclusive and exclusive
    if not isnan(valid_span[0]):
        with np.errstate(invalid='ignore'):
            if start_inclusive is True:
                flag_arr[inp < valid_span[0]] = QartodFlags.FAIL
            else:
                flag_arr[inp <= valid_span[0]] = QartodFlags.FAIL

    if not isnan(valid_span[1]):
        with np.errstate(invalid='ignore'):
            if end_inclusive is True:
                flag_arr[inp > valid_span[1]] = QartodFlags.FAIL
            else:
                flag_arr[inp >= valid_span[1]] = QartodFlags.FAIL

    # If the value is masked or nan set the flag to MISSING
    flag_arr[inp.mask] = QartodFlags.MISSING

    return flag_arr.reshape(original_shape)