#!/usr/bin/env python
# coding=utf-8
import io
import logging
import simplejson as json
from typing import Any, Union
from numbers import Real
from pyproj import Geod
from pathlib import Path
from datetime import date, datetime
from collections import OrderedDict as odict
from collections.abc import Mapping

import pandas as pd
import numpy as np
import pandas as pd
import xarray as xr
import geojson
from ruamel.yaml import YAML

N = Real
L = logging.getLogger(__name__)  # noqa


def add_flag_metadata(**kwargs):
    def wrapper(func : callable):
        for k, v in kwargs.items():
            setattr(func, k, v)
        return func
    return wrapper


def openf(p, **kwargs):
    """ Helper to allow one-line-lambdas to read file contents
    """
    with open(p, **kwargs) as f:
        return f.read()


def load_config_from_xarray(source):
    """Load an xarray dataset as a config dict
    """

    to_close = False
    if not isinstance(source, xr.Dataset):
        source = xr.open_dataset(source, decode_cf=False)
        to_close = True

    # If a global attribute exists, load as a YAML or JSON string and
    # ignore any config at the variable level
    if 'ioos_qc_config' in source.attrs:
        L.info("Using global attribute ioos_qc_config for QC config")
        return load_config_as_dict(source.attrs['ioos_qc_config'])

    # Iterate over variables and construct a config object
    y = odict()
    source = source.filter_by_attrs(
        ioos_qc_module=lambda x: x is not None,
        ioos_qc_test=lambda x: x is not None,
        ioos_qc_config=lambda x: x is not None,
        ioos_qc_target=lambda x: x is not None,
    )
    for dv in source.variables:
        if dv in source.dims:
            continue
        vobj = source[dv]

        # Because a data variables can have more than one check
        # associated with it we need to merge any existing configs
        # for this variable
        newdict = odict({
            vobj.ioos_qc_module: odict({
                vobj.ioos_qc_test: odict(json.loads(vobj.ioos_qc_config))
            })
        })
        merged = dict_update(
            y.get(vobj.ioos_qc_target, {}),
            newdict
        )
        y[vobj.ioos_qc_target] = merged

    # If we opened this xarray dataset from a file we should close it
    if to_close is True:
        source.close()

    return y


def load_config_as_dict(source : Union[str, dict, odict, Path, io.StringIO]
                        ) -> odict:
    """Load an object as a config dict. The source can be a dict, odict,
    YAML string, JSON string, a StringIO, or a file path to a valid YAML or JSON file.
    """
    yaml = YAML(typ='safe')
    if isinstance(source, odict):
        return source
    elif isinstance(source, dict):
        return odict(source)
    elif isinstance(source, xr.Dataset):
        return load_config_from_xarray(source)
    elif isinstance(source, (str, Path)):
        source = str(source)

        # Try to load as YAML, then JSON, then file path
        load_funcs = [
            lambda x: odict(yaml.load(x)),
            lambda x: odict(json.loads(x)),
            lambda x: load_config_from_xarray(x),
            lambda x: odict(yaml.load(openf(x))),  # noqa
            lambda x: odict(json.loads(openf(x))),  # noqa
        ]
        for lf in load_funcs:
            try:
                return lf(source)
            except BaseException:
                continue

    elif isinstance(source, io.StringIO):
        # Try to load as YAML, then JSON, then file path
        load_funcs = [
            lambda x: odict(yaml.load(x)),
            lambda x: odict(json.load(x)),
        ]
        for lf in load_funcs:
            try:
                return lf(source.getvalue())
            except BaseException:
                continue

    raise ValueError('Config source is not valid!')


def isfixedlength(lst : Union[list, tuple],
                  length : int
                  ) -> bool:
    if not isinstance(lst, (list, tuple)):
        raise ValueError('Required: list/tuple, Got: {}'.format(type(lst)))

    if len(lst) != length:
        raise ValueError(
            'Incorrect list/tuple length for {}. Required: {}, Got: {}'.format(
                lst,
                length,
                len(lst)
            )
        )

    return True


def isnan(v : Any) -> bool:
    return (
        v is None or
        v is np.nan or
        v is np.ma.masked
    )


def mapdates(dates):
    if hasattr(dates, 'dtype') and hasattr(dates.dtype, 'tz'):
        # pandas time objects with a datetime component, remove the timezone
        return dates.dt.tz_localize(None).astype('datetime64[ns]').to_numpy()
    elif hasattr(dates, 'dtype') and hasattr(dates, 'to_numpy'):
        # pandas time objects without a datetime component
        return dates.to_numpy().astype('datetime64[ns]')
    elif hasattr(dates, 'dtype') and np.issubdtype(dates.dtype, np.datetime64):
        # numpy datetime objects
        return dates.astype('datetime64[ns]')
    else:
        try:
            # Finally try unix epoch seconds
            return pd.to_datetime(dates, unit='s').values.astype('datetime64[ns]')
        except Exception:
            # strings work here but we don't advertise that
            return np.array(dates, dtype='datetime64[ns]')


def check_timestamps(times : np.ndarray,
                     max_time_interval : N = None
                     ) -> bool:
    """Sanity checks for timestamp arrays

    Checks that the times supplied are in monotonically increasing
    chronological order, and optionally that time intervals between
    measurements do not exceed a value `max_time_interval`.  Note that this is
    not a QARTOD test, but rather a utility test to make sure times are in the
    proper order and optionally do not have large gaps prior to processing the
    data.

    Args:
        times: Input array of timestamps
        max_time_interval: The interval between values should not exceed this
            value. [optional]
    """

    time_diff = np.diff(times)
    sort_diff = np.diff(sorted(times))
    # Check if there are differences between sorted and unsorted, and then
    # see if if there are any duplicate times.  Then check that none of the
    # diffs exceeds the sorted time.
    zero = np.array(0, dtype=time_diff.dtype)
    if not np.array_equal(time_diff, sort_diff) or np.any(sort_diff == zero):
        return False
    elif (max_time_interval is not None and
          np.any(sort_diff > max_time_interval)):
        return False
    else:
        return True


def dict_update(d : Mapping, u : Mapping) -> Mapping:
    # http://stackoverflow.com/a/3233356
    for k, v in u.items():
        if isinstance(d, Mapping):
            if isinstance(v, Mapping):
                r = dict_update(d.get(k, {}), v)
                d[k] = r
            else:
                d[k] = u[k]
        else:
            d = { k: u[k] }
    return d


def dict_depth(d):
    """ Get the depth of a dict
    """
    # https://stackoverflow.com/a/23499101
    if isinstance(d, dict):
        return 1 + (max(map(dict_depth, d.values())) if d else 0)
    return 0


def cf_safe_name(name : str) -> str:
    import re
    if isinstance(name, str):
        if re.match('^[0-9_]', name):
            # Add a letter to the front
            name = "v_{}".format(name)
        return re.sub(r'[^_a-zA-Z0-9]', "_", name)

    raise ValueError('Could not convert "{}" to a safe name'.format(name))


class GeoNumpyDateEncoder(geojson.GeoJSONEncoder):

    def default(self, obj : Any) -> Any:
        """If input object is an ndarray it will be converted into a list
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.generic):
            return obj.item()
        # elif isinstance(obj, pd.Timestamp):
        #     return obj.to_pydatetime().isoformat()
        elif isinstance(obj, (datetime, date)):
            return obj.isoformat()
        elif np.isnan(obj):
            return None

        return geojson.factory.GeoJSON.to_instance(obj)


def great_circle_distance(lat_arr, lon_arr):
    dist = np.ma.zeros(lon_arr.size, dtype=np.float64)
    g = Geod(ellps='WGS84')
    _, _, dist[1:] = g.inv(lon_arr[:-1], lat_arr[:-1], lon_arr[1:], lat_arr[1:])
    return dist
