#!/usr/bin/env python
# coding=utf-8
"""QC Config objects

Module to store the different QC modules in ioos_qc

Attributes:
    tw (namedtuple): The TimeWindow namedtuple definition
"""
import io
import logging
import re
import warnings
from pathlib import Path
from copy import deepcopy
from functools import partial
from typing import Union, List
from dataclasses import dataclass
from importlib import import_module
from inspect import getmodule, signature
from collections import namedtuple, defaultdict
from collections import OrderedDict as odict

from shapely.geometry import shape, GeometryCollection

from ioos_qc.results import collect_results
from ioos_qc.results import StreamConfigResult
from ioos_qc.utils import load_config_as_dict, dict_depth

L = logging.getLogger(__name__)  # noqa
ConfigTypes = Union[dict, odict, str, Path, io.StringIO]

tw = namedtuple('TimeWindow', ('starting', 'ending'), defaults=[None, None])


@dataclass(frozen=True)
class Context:
    window: tw = dataclass.field(default_factory=tw)
    region: GeometryCollection = dataclass.field(default=None)
    attrs: dict = dataclass.field(default_factory=dict)

    def __eq__(self, other):
        if isinstance(other, Context):
            return self.window == other.window and self.region == other.region
        return False

    def __repr__(self):
        return f'<Context window={self.window} region={self.region}>'


@dataclass(frozen=True)
class Call:
    stream_id: str
    call: partial
    context: Context = dataclass.field(default_factory=Context)
    attrs: dict = dataclass.field(default_factory=dict)

    @property
    def window(self):
        return self.context.window

    @property
    def region(self):
        return self.context.region

    @property
    def module(self) -> str:
        return self.call.func.__module__

    @property
    def method(self) -> str:
        return self.call.func.__name__

    @property
    def method_path(self) -> str:
        return f'{self.module}.{self.method}'

    @property
    def args(self) -> tuple:
        return self.call.args

    @property
    def kwargs(self) -> dict:
        return self.call.keywords

    def __key__(self):
        return (
            self.stream_id,
            self.context,
            self.module,
            self.method,
            self.args,
            tuple(self.kwargs.items())
        )

    def __hash__(self):
        return hash(self.__key__())

    def __eq__(self, other):
        if isinstance(other, Call):
            return self.__key__() == other.__key__()
        return NotImplemented

    def __repr__(self):
        ret = f'<Call stream_id={self.stream_id}'
        if self.context.window.starting:
            ret += f' starting={self.window.starting}'
        if self.context.window.ending:
            ret += f' ending={self.window.ending}'
        if self.context.region is not None:
            return ' region=True'
        ret += f' {self.module}.{self.method}({self.args}, {self.kwargs})>'
        return ret

    def run(self, **passedkwargs):
        aggregates = []
        results = []
        # results = defaultdict(odict)

        for runfunc, kwargs in self.methods.items():

            # This is just easier to query for them rather than set in the self.methods
            # method. Just a bit of python magic to try and simplify.
            testname = runfunc.__name__
            package = getmodule(runfunc).__name__.split('.')[1]

            # Skip any aggregate flags and run them at the end
            if getattr(runfunc, 'aggregate', False) is True:
                L.debug("Skipping aggregate (will run after all other tests)")
                aggregates.append((package, testname, runfunc))
                continue

            # Get our own copy of the kwargs object so we can change it
            testkwargs = deepcopy(passedkwargs)
            # Merges dicts
            testkwargs = odict({ **kwargs, **testkwargs })

            # Get the arguments that the test functions support
            sig = signature(runfunc)
            valid_keywords = [
                p.name for p in sig.parameters.values()
                if p.kind == p.POSITIONAL_OR_KEYWORD
            ]
            testkwargs = {
                k: v for k, v in testkwargs.items()
                if k in valid_keywords
            }
            try:
                results.append(
                    StreamConfigResult(
                        package=package,
                        test=testname,
                        function=runfunc,
                        results=runfunc(**testkwargs)
                    )
                )
                # results[package][testname] = runfunc(**testkwargs)
            except Exception as e:
                L.error(f'Could not run "{package}.{testname}: {e}')
                continue

        # Now run the aggregates at the end when all other results are complete
        for aggpackage, aggtestname, aggfunc in aggregates:
            results.append(
                StreamConfigResult(
                    package=aggpackage,
                    test=aggtestname,
                    function=aggfunc,
                    results=aggfunc(results)
                )
            )
            # results[aggpackage][aggtestname] = aggfunc(results, functions=tests_run)

        return results

def extract_calls(source) -> List[Call]:
    """
    Extracts call objects from a source object

    Args:
        source ([any]): The source of Call objects, this can be a:
            * Call object
            * list of Call objects
            * list of objects with the 'calls' attribute
            * NewConfig object
            * Object with the 'calls' attribute

    Returns:
        List[Call]: List of extracted Call objects

    """
    if isinstance(source, Call):
        return [source]
    elif isinstance(source, (tuple, list)):
        # list of Call objects
        calls = [ c for c in source if isinstance(c, Call) ]
        # list of objects with the 'calls' attribute
        [
            calls.extend([
                x for x in c.calls if isinstance(x, Call)
            ])
            for c in source if hasattr(c, 'calls')
        ]
        return calls
    elif isinstance(Config):
        # Config object
        return source.calls
    elif hasattr(source, 'calls'):
        # Object with the 'calls' attribute
        return source.calls
    return []


class Config:
    """ A class to load any ioos_qc configuration setup into a list of callable objects
    that will run quality checks. The resulting list of quality checks parsed from a config
    file can be appended and edited until they are ready to be run. On run the checks are
    consolidated into an efficient structure for indexing the dataset (stream) it is run against
    so things like subsetting by time and space only happen once for each test in the same Context.

    How the individual checks are collected is up to each individual Stream implementation, this
    class only pares various formats and versions of a config into a list of Call objects.
    """

    def __init__(self, source, version=None):
        """
        Args:
            source: The QC configuration representation in one of the following formats:
                python dict or odict
                JSON/YAML filepath (str or Path object)
                JSON/YAML str
                JSON/YAML StringIO
                netCDF4/xarray filepath
                netCDF4/xarray Dataset
                list of Call objects
        """
        # A fully encapsulated Call objects that are configured
        # There are later grouped by window/region to more efficiently process
        # groups of indexes and variables
        self._calls = []

        # If we are passed an object we can extract calls from do so
        # Else, process as a Config object
        extracted = extract_calls(source)
        if extracted:
            self._calls = extracted
        else:
            # Parse config based on version
            # Massage and return the correct type of config object depending on the input
            self.config = load_config_as_dict(source)
            if 'contexts' in self.config:
                # Return a list of ContextConfig
                for c in self.config['contexts']:
                    self._calls.extend(list(ContextConfig(c).calls))
            elif 'streams' in self.config:
                # Return a list with just one ContextConfig
                self._calls += list(ContextConfig(self.config).calls)
            elif dict_depth(self.config) >= 4:
                # This is a StreamConfig
                self._calls += list(ContextConfig(odict(streams=self.config)).calls)
            else:
                # This is a QC Config
                raise ValueError("Can not add context to a QC Config object. Create it manually.")

    @property
    def calls(self):
        return self._calls

    def has(self, stream_id : str, method: Union[callable, str]):
        if isinstance(method, str):
            for c in self._calls:
                if c.stream_id == stream_id and c.method_path == method:
                    return c
        elif isinstance(method, callable):
            for c in self._calls:
                if (c.stream_id == stream_id and
                    c.method == method.__module__ and
                    c.method == method.__name__):
                    return c
        return False

    def add(self, source) -> None:
        """
        Adds a source of calls to this Config. See extract_calls for information on the
        types of objects accepted as the source parameter. The changes the internal .calls
        attribute and returns None.

        Args:
            source ([any]): The source of Call objects, this can be a:
                * Call object
                * list of Call objects
                * list of objects with the 'calls' attribute
                * Config object
                * Object with the 'calls' attribute
        """
        extracted = extract_calls(source)
        self._calls += extracted


class ContextConfig:
    """A collection of a Region, a TimeWindow and a list of StreamConfig objects

    Defines a set of quality checks to run against multiple input streams.
    This can include a region and a time window to subset any DataStreams by before running checks.

    region: None
    window:
        starting: 2020-01-01T00:00:00Z
        ending: 2020-04-01T00:00:00Z
    streams:
        variable1:    # stream_id
            qartod:   # StreamConfig
                location_test:
                    bbox: [-80, 40, -70, 60]
        variable2:    # stream_id
            qartod:   # StreamConfig
                gross_range_test:
                    suspect_span: [1, 11]
                    fail_span: [0, 12]

    Helper methods exist to run this check against a different inputs:
        * pandas.DataFrame, dask.DataFrame, netCDF4.Dataset, xarray.Dataset, ERDDAP URL

    Attributes:
        config (odict): dict representation of the parsed ContextConfig source
        region (GeometryCollection): A `shapely` object representing the valid geographic region
        window (namedtuple): A TimeWindow object representing the valid time period
        streams (odict): dict represenatation of the parsed StreamConfig objects
    """

    def __init__(self, source: ConfigTypes):
        self.config = load_config_as_dict(source)

        # Region
        self._calls = []
        self.region = None
        self.attrs = self.config.get('attrs', {})

        if 'region' in self.config:
            # Convert region to a GeometryCollection Shapely object.
            # buffer(0) is a trick for fixing scenarios where polygons have overlapping coordinates
            # https://medium.com/@pramukta/recipe-importing-geojson-into-shapely-da1edf79f41d
            if isinstance(self.config['region'], GeometryCollection):
                self.region = self.config['region']
            elif self.config['region'] and 'features' in self.config['region']:
                # Feature based GeoJSON
                self.region = GeometryCollection([
                    shape(feature['geometry']).buffer(0) for feature in self.config['region']['features']
                ])
            elif self.config['region'] and 'geometry' in self.config['region']:
                # Geometry based GeoJSON
                self.region = GeometryCollection([
                    shape(self.config['region']['geometry']).buffer(0)
                ])
            else:
                L.warning('Ignoring region because it could not be parsed, is it valid GeoJSON?')

        # Window
        if 'window' in self.config and isinstance(self.config['window'], tw):
            self.window = self.config['window']
        elif 'window' in self.config:
            self.window = tw(**self.config['window'])

        self.context = Context(
            window=self.window,
            region=self.region,
            attrs=self.attrs
        )

        # Extract each Call from the nested JSON
        """
        Calls
        This parses through available checks and selects the actual test functions
        to run, but doesn't actually run anything. It just sets up the object to be
        run later by iterating over the configs.
        """
        for stream_id, sc in self.config['streams'].items():

            for package, modules in sc.items():
                try:
                    testpackage = import_module('ioos_qc.{}'.format(package))
                except ImportError:
                    L.warning(f'No ioos_qc package "{package}" was found, skipping.')
                    continue

                for testname, kwargs in modules.items():
                    kwargs = kwargs or {}
                    if not hasattr(testpackage, testname):
                        L.warning(f'No ioos_qc method "{package}.{testname}" was found, skipping')
                        continue
                    else:
                        runfunc = getattr(testpackage, testname)

                    self._calls.append(
                        Call(
                            stream_id=stream_id,
                            context=self.context,
                            call=partial(runfunc, (), **kwargs),
                            attrs=getattr(sc, 'attrs', {})
                        )
                    )

    @property
    def calls(self):
        return self._calls

    def add(self, source) -> None:
        """
        Adds a source of calls to this ContextConfig. See extract_calls for information on the
        types of objects accepted as the source parameter. The changes the internal .calls
        attribute and returns None.

        Args:
            source ([any]): The source of Call objects, this can be a:
                * Call object
                * list of Call objects
                * list of objects with the 'calls' attribute
                * Config object
                * Object with the 'calls' attribute
        """
        extracted = extract_calls(source)
        self._calls.extend([ e for e in extracted if e.context == self.context ])

    def __str__(self):
        sc = list(self.streams.keys())
        return (
            f"<ContextConfig "
            f"calls={len(self._calls)} "
            f"region={self.region is not None} "
            f"window={self.window.starting is not None or self.window.ending is not None}"
            ">"
        )

    def __repr__(self):
        return self.__str__()