#!/usr/bin/env python
# coding=utf-8
import logging
import unittest

import numpy as np
import pandas as pd
import xarray as xr
import numpy.testing as npt

from ioos_qc.streams import (
    Config,
    NetcdfStream,
    NumpyStream,
    PandasStream,
    XarrayStream,
)
from ioos_qc.results import collect_results

L = logging.getLogger('ioos_qc')
L.setLevel(logging.INFO)
L.handlers = [logging.StreamHandler()]


class PandasStreamTest(unittest.TestCase):
    def setUp(self):

        config = """
            region: something
            window:
                starting: 2020-01-01T00:00:00Z
                ending: 2020-04-01T00:00:00Z
            streams:
                variable1:
                    qartod:
                        gross_range_test:
                            suspect_span: [20, 30]
                            fail_span: [10, 40]
        """
        self.config = Config(config)

        rows = 50
        data_inputs = {
            'time': pd.date_range(start='01/01/2020', periods=rows, freq='D'),
            'z': 2.0,
            'lat': 36.1,
            'lon': -76.5,
            'variable1': np.arange(0, rows),
        }
        self.df = pd.DataFrame(data_inputs)

    def test_run(self):
        ps = PandasStream(self.df)
        raw_results = ps.run(self.config)
        results = collect_results(raw_results, how='dict')

        # First ten (0-9 values) fail
        npt.assert_array_equal(
            results['variable1']['qartod']['gross_range_test'][0:10],
            np.array([4, 4, 4, 4, 4, 4, 4, 4, 4, 4])
        )

        # Next ten (10-19 values) suspect
        npt.assert_array_equal(
            results['variable1']['qartod']['gross_range_test'][10:20],
            np.array([3, 3, 3, 3, 3, 3, 3, 3, 3, 3])
        )
        # Next ten (20-29 values) pass
        npt.assert_array_equal(
            results['variable1']['qartod']['gross_range_test'][20:30],
            np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        )
        # Next ten, first value (30) pass becuasethe test is inclusive
        # and (31-39 values) suspect
        npt.assert_array_equal(
            results['variable1']['qartod']['gross_range_test'][30:40],
            np.array([1, 3, 3, 3, 3, 3, 3, 3, 3, 3])
        )
        # Next ten, first value (40) suspect becuasethe test is inclusive
        # and (41-49 values) fail
        npt.assert_array_equal(
            results['variable1']['qartod']['gross_range_test'][40:50],
            np.array([3, 4, 4, 4, 4, 4, 4, 4, 4, 4])
        )


class NumpyStreamTestLightConfig(unittest.TestCase):
    def setUp(self):

        config = """
            streams:
                variable1:
                    qartod:
                        gross_range_test:
                            suspect_span: [20, 30]
                            fail_span: [10, 40]
        """
        self.config = Config(config)

        rows = 50
        self.tinp = pd.date_range(start='01/01/2020', periods=rows, freq='D').values
        self.zinp = np.full_like(self.tinp, 2.0)
        self.lat = np.full_like(self.tinp, 36.1)
        self.lon = np.full_like(self.tinp, -76.5)

    def test_run(self):
        # Input is the values 0-49, easy testing
        inp = np.arange(0, self.tinp.size)

        ns = NumpyStream(inp, self.tinp, self.zinp, self.lat, self.lon)
        results = ns.run(self.config)
        results = collect_results(results, how='dict')

        # First ten (0-9 values) fail
        npt.assert_array_equal(
            results['variable1']['qartod']['gross_range_test'][0:10],
            np.array([4, 4, 4, 4, 4, 4, 4, 4, 4, 4])
        )
        # Next ten (10-19 values) suspect
        npt.assert_array_equal(
            results['variable1']['qartod']['gross_range_test'][10:20],
            np.array([3, 3, 3, 3, 3, 3, 3, 3, 3, 3])
        )
        # Next ten (20-29 values) pass
        npt.assert_array_equal(
            results['variable1']['qartod']['gross_range_test'][20:30],
            np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        )
        # Next ten, first value (30) pass because the test is inclusive
        # and (31-39 values) suspect
        npt.assert_array_equal(
            results['variable1']['qartod']['gross_range_test'][30:40],
            np.array([1, 3, 3, 3, 3, 3, 3, 3, 3, 3])
        )
        # Next ten, first value (40) suspect because the test is inclusive
        # and (41-49 values) fail
        npt.assert_array_equal(
            results['variable1']['qartod']['gross_range_test'][40:50],
            np.array([3, 4, 4, 4, 4, 4, 4, 4, 4, 4])
        )


class NumpyStreamTest(unittest.TestCase):
    def setUp(self):

        config = """
            region: something
            window:
                starting: 2020-01-01T00:00:00Z
                ending: 2020-04-01T00:00:00Z
            streams:
                variable1:
                    qartod:
                        gross_range_test:
                            suspect_span: [20, 30]
                            fail_span: [10, 40]
        """
        self.config = Config(config)

        rows = 50
        self.tinp = pd.date_range(start='01/01/2020', periods=rows, freq='D').values
        self.zinp = np.full_like(self.tinp, 2.0)
        self.lat = np.full_like(self.tinp, 36.1)
        self.lon = np.full_like(self.tinp, -76.5)

    def test_run(self):
        # Input is the values 0-49, easy testing
        inp = np.arange(0, self.tinp.size)

        ns = NumpyStream(inp, self.tinp, self.zinp, self.lat, self.lon)
        results = ns.run(self.config)
        results = collect_results(results, how='dict')

        # First ten (0-9 values) fail
        npt.assert_array_equal(
            results['variable1']['qartod']['gross_range_test'][0:10],
            np.array([4, 4, 4, 4, 4, 4, 4, 4, 4, 4])
        )
        # Next ten (10-19 values) suspect
        npt.assert_array_equal(
            results['variable1']['qartod']['gross_range_test'][10:20],
            np.array([3, 3, 3, 3, 3, 3, 3, 3, 3, 3])
        )
        # Next ten (20-29 values) pass
        npt.assert_array_equal(
            results['variable1']['qartod']['gross_range_test'][20:30],
            np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        )
        # Next ten, first value (30) pass because the test is inclusive
        # and (31-39 values) suspect
        npt.assert_array_equal(
            results['variable1']['qartod']['gross_range_test'][30:40],
            np.array([1, 3, 3, 3, 3, 3, 3, 3, 3, 3])
        )
        # Next ten, first value (40) suspect because the test is inclusive
        # and (41-49 values) fail
        npt.assert_array_equal(
            results['variable1']['qartod']['gross_range_test'][40:50],
            np.array([3, 4, 4, 4, 4, 4, 4, 4, 4, 4])
        )


class NetcdfStreamTest(unittest.TestCase):
    def setUp(self):

        config = """
            region: something
            window:
                starting: 2020-01-01T00:00:00Z
                ending: 2020-04-01T00:00:00Z
            streams:
                variable1:
                    qartod:
                        gross_range_test:
                            suspect_span: [20, 30]
                            fail_span: [10, 40]
        """
        self.config = Config(config)

        rows = 50
        data_inputs = {
            'time': pd.date_range(start='01/01/2020', periods=rows, freq='D'),
            'z': 2.0,
            'lat': 36.1,
            'lon': -76.5,
            'variable1': np.arange(0, rows),
        }
        df = pd.DataFrame(data_inputs)
        self.ds = xr.Dataset.from_dataframe(df)

    def tearDown(self):
        self.ds.close()

    def test_run(self):
        ns = NetcdfStream(self.ds)
        results = ns.run(self.config)
        results = collect_results(results, how='dict')

        # First ten (0-9 values) fail
        npt.assert_array_equal(
            results['variable1']['qartod']['gross_range_test'][0:10],
            np.array([4, 4, 4, 4, 4, 4, 4, 4, 4, 4])
        )
        # Next ten (10-19 values) suspect
        npt.assert_array_equal(
            results['variable1']['qartod']['gross_range_test'][10:20],
            np.array([3, 3, 3, 3, 3, 3, 3, 3, 3, 3])
        )
        # Next ten (20-29 values) pass
        npt.assert_array_equal(
            results['variable1']['qartod']['gross_range_test'][20:30],
            np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        )
        # Next ten, first value (30) pass because the test is inclusive
        # and (31-39 values) suspect
        npt.assert_array_equal(
            results['variable1']['qartod']['gross_range_test'][30:40],
            np.array([1, 3, 3, 3, 3, 3, 3, 3, 3, 3])
        )
        # Next ten, first value (40) suspect because the test is inclusive
        # and (41-49 values) fail
        npt.assert_array_equal(
            results['variable1']['qartod']['gross_range_test'][40:50],
            np.array([3, 4, 4, 4, 4, 4, 4, 4, 4, 4])
        )


class XarrayStreamTest(unittest.TestCase):
    def setUp(self):

        config = """
            region: something
            window:
                starting: 2020-01-01T00:00:00Z
                ending: 2020-04-01T00:00:00Z
            streams:
                variable1:
                    qartod:
                        gross_range_test:
                            suspect_span: [20, 30]
                            fail_span: [10, 40]
        """
        self.config = Config(config)

        rows = 50
        data_inputs = {
            'time': pd.date_range(start='01/01/2020', periods=rows, freq='D'),
            'z': 2.0,
            'lat': 36.1,
            'lon': -76.5,
            'variable1': np.arange(0, rows),
        }
        df = pd.DataFrame(data_inputs).set_index('time')
        self.ds = xr.Dataset.from_dataframe(df)

    def tearDown(self):
        self.ds.close()

    def test_run(self):
        xs = XarrayStream(self.ds)
        results = xs.run(self.config)
        results = collect_results(results, how='dict')

        # First ten (0-9 values) fail
        npt.assert_array_equal(
            results['variable1']['qartod']['gross_range_test'][0:10],
            np.array([4, 4, 4, 4, 4, 4, 4, 4, 4, 4])
        )
        # Next ten (10-19 values) suspect
        npt.assert_array_equal(
            results['variable1']['qartod']['gross_range_test'][10:20],
            np.array([3, 3, 3, 3, 3, 3, 3, 3, 3, 3])
        )
        # Next ten (20-29 values) pass
        npt.assert_array_equal(
            results['variable1']['qartod']['gross_range_test'][20:30],
            np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        )
        # Next ten, first value (30) pass because the test is inclusive
        # and (31-39 values) suspect
        npt.assert_array_equal(
            results['variable1']['qartod']['gross_range_test'][30:40],
            np.array([1, 3, 3, 3, 3, 3, 3, 3, 3, 3])
        )
        # Next ten, first value (40) suspect because the test is inclusive
        # and (41-49 values) fail
        npt.assert_array_equal(
            results['variable1']['qartod']['gross_range_test'][40:50],
            np.array([3, 4, 4, 4, 4, 4, 4, 4, 4, 4])
        )


class XarrayStreamManyContextTest(unittest.TestCase):
    def setUp(self):
        config = """
            contexts:
                -   region: something
                    window:
                        starting: 2020-01-01T00:00:00Z
                        ending: 2020-02-01T00:00:00Z
                    streams:
                        variable1:
                            qartod:
                                gross_range_test:
                                    suspect_span: [3, 4]
                                    fail_span: [2, 5]
                        variable2:
                            qartod:
                                gross_range_test:
                                    suspect_span: [23, 24]
                                    fail_span: [22, 25]
                -   region: something else
                    window:
                        starting: 2020-02-01T00:00:00Z
                        ending: 2020-03-01T00:00:00Z
                    streams:
                        variable1:
                            qartod:
                                gross_range_test:
                                    suspect_span: [43, 44]
                                    fail_span: [42, 45]
                        variable2:
                            qartod:
                                gross_range_test:
                                    suspect_span: [23, 24]
                                    fail_span: [22, 25]
        """
        self.config = Config(config)

        rows = 50
        self.vardata = np.arange(0, rows)
        data_inputs = {
            'time': pd.date_range(start='01/01/2020', periods=rows, freq='D'),
            'z': 2.0,
            'lat': 36.1,
            'lon': -76.5,
            'variable1': self.vardata,
            'variable2': self.vardata,
        }
        df = pd.DataFrame(data_inputs).set_index('time')
        self.ds = xr.Dataset.from_dataframe(df)

    def tearDown(self):
        self.ds.close()

    def test_run_dict_results(self):
        xs = XarrayStream(self.ds)
        results = xs.run(self.config)
        results = collect_results(results, how='dict')

        # Variable 1
        npt.assert_array_equal(
            results['variable1']['qartod']['gross_range_test'][0:8],
            np.array([4, 4, 3, 1, 1, 3, 4, 4])
        )
        npt.assert_array_equal(
            results['variable1']['qartod']['gross_range_test'][8:40],
            np.full((32,), 4)
        )
        npt.assert_array_equal(
            results['variable1']['qartod']['gross_range_test'][40:48],
            np.array([4, 4, 3, 1, 1, 3, 4, 4])
        )

        # Variable 2
        npt.assert_array_equal(
            results['variable2']['qartod']['gross_range_test'][0:20],
            np.full((20,), 4)
        )
        npt.assert_array_equal(
            results['variable2']['qartod']['gross_range_test'][20:28],
            np.array([4, 4, 3, 1, 1, 3, 4, 4])
        )
        npt.assert_array_equal(
            results['variable2']['qartod']['gross_range_test'][28:50],
            np.full((22,), 4)
        )


class PandasStreamManyContextTest(unittest.TestCase):
    def setUp(self):
        config = """
            contexts:
                -   region: something
                    window:
                        starting: 2020-01-01T00:00:00Z
                        ending: 2020-02-01T00:00:00Z
                    streams:
                        variable1:
                            qartod:
                                gross_range_test:
                                    suspect_span: [3, 4]
                                    fail_span: [2, 5]
                        variable2:
                            qartod:
                                gross_range_test:
                                    suspect_span: [23, 24]
                                    fail_span: [22, 25]
                -   region: something else
                    window:
                        starting: 2020-02-01T00:00:00Z
                        ending: 2020-03-01T00:00:00Z
                    streams:
                        variable1:
                            qartod:
                                gross_range_test:
                                    suspect_span: [43, 44]
                                    fail_span: [42, 45]
                        variable2:
                            qartod:
                                gross_range_test:
                                    suspect_span: [23, 24]
                                    fail_span: [22, 25]
        """
        self.config = Config(config)

        rows = 50
        self.vardata = np.arange(0, rows)
        data_inputs = {
            'time': pd.date_range(start='01/01/2020', periods=rows, freq='D'),
            'z': 2.0,
            'lat': 36.1,
            'lon': -76.5,
            'variable1': self.vardata,
            'variable2': self.vardata,
        }
        self.df = pd.DataFrame(data_inputs)

    def test_run_dict_results(self):
        ps = PandasStream(self.df)
        results = ps.run(self.config)
        results = collect_results(results, how='dict')

        # Variable 1
        npt.assert_array_equal(
            results['variable1']['qartod']['gross_range_test'][0:8],
            np.array([4, 4, 3, 1, 1, 3, 4, 4])
        )
        npt.assert_array_equal(
            results['variable1']['qartod']['gross_range_test'][8:40],
            np.full((32,), 4)
        )
        npt.assert_array_equal(
            results['variable1']['qartod']['gross_range_test'][40:48],
            np.array([4, 4, 3, 1, 1, 3, 4, 4])
        )

        # Variable 2
        npt.assert_array_equal(
            results['variable2']['qartod']['gross_range_test'][0:20],
            np.full((20,), 4)
        )
        npt.assert_array_equal(
            results['variable2']['qartod']['gross_range_test'][20:28],
            np.array([4, 4, 3, 1, 1, 3, 4, 4])
        )
        npt.assert_array_equal(
            results['variable2']['qartod']['gross_range_test'][28:50],
            np.full((22,), 4)
        )

    def test_run_list_results(self):
        ps = PandasStream(self.df)
        results = ps.run(self.config)
        results = collect_results(results, how='list')

        var1_gr = next(res for res in results if res.stream_id == 'variable1' and res.test == 'gross_range_test')
        var2_gr = next(res for res in results if res.stream_id == 'variable2' and res.test == 'gross_range_test')
        # Variable 1
        # Actual data returned in full
        npt.assert_array_equal(
            var1_gr.data,
            self.vardata
        )
        # QC tests
        npt.assert_array_equal(
            var1_gr.results[0:8],
            np.array([4, 4, 3, 1, 1, 3, 4, 4])
        )
        npt.assert_array_equal(
            var1_gr.results[8:40],
            np.full((32,), 4)
        )
        npt.assert_array_equal(
            var1_gr.results[40:48],
            np.array([4, 4, 3, 1, 1, 3, 4, 4])
        )

        # Variable 2
        npt.assert_array_equal(
            var2_gr.results[0:20],
            np.full((20,), 4)
        )
        npt.assert_array_equal(
            var2_gr.results[20:28],
            np.array([4, 4, 3, 1, 1, 3, 4, 4])
        )
        npt.assert_array_equal(
            var2_gr.results[28:50],
            np.full((22,), 4)
        )
