#!/usr/bin/env python
# coding=utf-8
import logging
import unittest
import warnings

import numpy as np
import pandas as pd
import numpy.testing as npt

from ioos_qc import qartod as qartod

L = logging.getLogger('ioos_qc')
L.setLevel(logging.INFO)
L.handlers = [logging.StreamHandler()]


def dask_arr(vals):
    """
    If dask is enabled for this environment, return dask array of values. Otherwise, return values.
    """
    try:
        import dask.array as da
        return da.from_array(vals, chunks=2)
    except ImportError:
        return vals


class QartodLocationTest(unittest.TestCase):

    def test_location(self):
        """
        Ensure that longitudes and latitudes are within reasonable bounds.
        """
        lon = [  80.0, -78.5, 500.500]
        lat = [np.NaN,  50.0,   -60.0]

        npt.assert_array_equal(
            qartod.location_test(lon=lon, lat=lat),
            np.ma.array([4, 1, 4])
        )

        lon = np.array(lon)
        lat = np.array(lat)
        npt.assert_array_equal(
            qartod.location_test(lon=lon, lat=lat),
            np.ma.array([4, 1, 4])
        )

        lon = dask_arr(lon)
        lat = dask_arr(lat)
        npt.assert_array_equal(
            qartod.location_test(lon=lon, lat=lat),
            np.ma.array([4, 1, 4])
        )

    def test_single_location_none(self):
        # Masked/None/NaN values return "UNKNOWN"
        lon = [None]
        lat = [None]
        npt.assert_array_equal(
            qartod.location_test(lon=lon, lat=lat),
            np.ma.array([9])
        )

        lon = np.array(lon)
        lat = np.array(lat)
        npt.assert_array_equal(
            qartod.location_test(lon=lon, lat=lat),
            np.ma.array([9])
        )

        lon = dask_arr(lon)
        lat = dask_arr(lat)
        npt.assert_array_equal(
            qartod.location_test(lon=lon, lat=lat),
            np.ma.array([9])
        )

    def test_single_location_nan(self):
        lon = [np.nan]
        lat = [np.nan]
        npt.assert_array_equal(
            qartod.location_test(lon=lon, lat=lat),
            np.ma.array([9])
        )

        lon = np.array(lon)
        lat = np.array(lat)
        npt.assert_array_equal(
            qartod.location_test(lon=lon, lat=lat),
            np.ma.array([9])
        )

        # Test dask nan input
        lon = dask_arr(lon)
        lat = dask_arr(lat)
        npt.assert_array_equal(
            qartod.location_test(lon=lon, lat=lat),
            np.ma.array([9])
        )

    def test_location_bad_input(self):
        # Wrong type lon
        with self.assertRaises(ValueError):
            qartod.location_test(lon='hello', lat=70)

        # Wrong type lat
        with self.assertRaises(ValueError):
            qartod.location_test(lon=70, lat='foo')

        # Wrong type bbox
        with self.assertRaises(ValueError):
            qartod.location_test(lon=70, lat=70, bbox='hi')

        # Wrong size bbox
        with self.assertRaises(ValueError):
            qartod.location_test(lon=70, lat=70, bbox=(1, 2))

    def test_location_bbox(self):
        lon = [80,   -78, -71, -79, 500]
        lat = [None,  50,  59,  10, -60]
        npt.assert_array_equal(
            qartod.location_test(lon=lon, lat=lat, bbox=[-80, 40, -70, 60]),
            np.ma.array([4, 1, 1, 4, 4])
        )

        lon = np.asarray([80,   -78, -71, -79, 500], dtype=np.float64)
        lat = np.asarray([None,  50,  59,  10, -60], dtype=np.float64)
        npt.assert_array_equal(
            qartod.location_test(lon=lon, lat=lat, bbox=[-80, 40, -70, 60]),
            np.ma.array([4, 1, 1, 4, 4])
        )

        lon = dask_arr(np.asarray([80,   -78, -71, -79, 500], dtype=np.float64))
        lat = dask_arr(np.asarray([None,  50,  59,  10, -60], dtype=np.float64))
        npt.assert_array_equal(
            qartod.location_test(lon=lon, lat=lat, bbox=[-80, 40, -70, 60]),
            np.ma.array([4, 1, 1, 4, 4])
        )

    def test_location_distance_threshold(self):
        """
        Tests a user defined distance threshold between successive points.
        """
        lon = np.array([-71.05, -71.06, -80.0])
        lat = np.array([41.0, 41.02, 45.05])

        npt.assert_array_equal(
            qartod.location_test(lon, lat),
            np.array([1, 1, 1])
        )
        npt.assert_array_equal(
            qartod.location_test(lon, lat, range_max=3000.0),
            np.ma.array([1, 1, 3])
        )


class QartodGrossRangeTest(unittest.TestCase):

    def test_gross_range_check(self):
        """See if user and sensor ranges are picked up."""
        fail_span = (10, 50)
        suspect_span = (20, 40)
        vals = [
            5, 10,               # Sensor range.
            15,                  # User range.
            20, 25, 30, 35, 40,  # Valid
            45,                  # User range.
            51                   # Sensor range.
        ]
        result = np.ma.array([
            4, 3,
            3,
            1, 1, 1, 1, 1,
            3,
            4
        ])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            inputs = [
                vals,
                np.array(vals, dtype=np.integer),
                np.array(vals, dtype=np.float64),
                dask_arr(np.array(vals, dtype=np.integer)),
                dask_arr(np.array(vals, dtype=np.float64))
            ]

        for i in inputs:
            npt.assert_array_equal(
                qartod.gross_range_test(
                    inp=i,
                    fail_span=fail_span,
                    suspect_span=suspect_span
                ),
                result
            )

    def test_gross_range_bad_input(self):
        with self.assertRaises(ValueError):
            qartod.gross_range_test(
                inp=np.array([5]),
                fail_span=10,
                suspect_span=(1, 1)
            )

        with self.assertRaises(ValueError):
            qartod.gross_range_test(
                inp=np.array([5]),
                fail_span=(1, 1),
                suspect_span=10
            )

        with self.assertRaises(ValueError):
            qartod.gross_range_test(
                inp=np.array([5]),
                fail_span=(1, 1),
                suspect_span=(2, 2)
            )

    def test_gross_range_check_masked(self):
        """See if user and sensor ranges are picked up."""
        fail_span = (10, 50)
        suspect_span = (20, 40)
        vals = [
            None,                # None
            10,                  # Sensor range.
            15,                  # User range.
            20, 25, 30, 35, 40,  # Valid
            np.nan,              # np.nan
            51,                  # Sensor range.
            np.ma.masked         # np.ma.masked
        ]
        result = np.ma.array([
            9,
            3,
            3,
            1, 1, 1, 1, 1,
            9,
            4,
            9
        ])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            inputs = [
                vals,
                np.array(vals, dtype=np.float64),
                dask_arr(np.array(vals, dtype=np.float64))
            ]

        for i in inputs:
            npt.assert_array_equal(
                qartod.gross_range_test(
                    vals,
                    fail_span,
                    suspect_span
                ),
                result

            )


class QartodClimatologyPeriodTest(unittest.TestCase):

    def _run_test(self, tspan, period):
        cc = qartod.ClimatologyConfig()
        cc.add(
            vspan=(10, 20),  # range of valid values
            tspan=tspan,  # jan, feb and march
            period=period
        )
        test_inputs = [
            # in feb, but outside range of valid values
            (
                np.datetime64('2011-02-02'),
                9,
                None
            ),
            # in feb, and within range of valid values
            (
                np.datetime64('2013-02-03'),
                11,
                None
            ),
            # Not run, outside of time range
            (
                np.datetime64('2011-04-01'),
                21,
                None
            ),
            # Not run, outside of time range
            (
                np.datetime64('2011-12-31'),
                21,
                None
            ),
            # leap day, with valid values
            (
                np.datetime64('2020-02-29'),
                15,
                None
            ),
        ]
        times, values, depths = zip(*test_inputs)
        inputs = [
            values,
            np.asarray(values, dtype=np.float64),
            dask_arr(np.asarray(values, dtype=np.float64))
        ]

        for i in inputs:
            results = qartod.climatology_test(
                config=cc,
                tinp=times,
                inp=i,
                zinp=depths
            )
            npt.assert_array_equal(
                results,
                np.ma.array([3, 1, 2, 2, 1])
            )

    def test_climatology_test_periods_monthly(self):
        self._run_test((0, 3), 'month')

    def test_climatology_test_periods_week_of_year(self):
        self._run_test((0, 12), 'weekofyear')

    def test_climatology_test_periods_day_of_year(self):
        self._run_test((0, 90), 'dayofyear')

    def test_climatology_test_periods_quarter(self):
        self._run_test((0, 1), 'quarter')


class QartodClimatologyPeriodFullCoverageTest(unittest.TestCase):
    # Test that we can define climatology periods across the whole year,
    # and test data ranges across several years

    def setUp(self):
        self.tinp = list(pd.date_range(start='2018-01-01', end='2020-12-31', freq='D'))
        self.values = np.ones(len(self.tinp))
        self.zinp = np.zeros(len(self.tinp))

    def _run_test(self, cc):
        # just run test and make sure we don't get any errors
        qartod.climatology_test(
            config=cc,
            tinp=self.tinp,
            inp=self.values,
            zinp=self.zinp
        )

    def test_quarterly_periods(self):
        vspan = (10, 20)
        cc = qartod.ClimatologyConfig()
        cc.add(
            tspan=(0, 1),       # Q1
            period='quarter',
            vspan=vspan,
        )
        cc.add(
            tspan=(1, 3),       # Q2-Q3
            period='quarter',
            vspan=vspan,
        )
        cc.add(
            tspan=(3, 4),       # Q4
            period='quarter',
            vspan=vspan,
        )
        self._run_test(cc)

    def test_monthly_periods(self):
        vspan = (10, 20)
        cc = qartod.ClimatologyConfig()
        cc.add(
            tspan=(0, 1),       # jan
            period='month',
            vspan=vspan,
        )
        cc.add(
            tspan=(1, 2),       # feb
            period='month',
            vspan=vspan,
        )
        cc.add(
            tspan=(2, 3),       # mar
            period='month',
            vspan=vspan,
        )
        cc.add(
            tspan=(3, 10),       # apr-nov
            period='month',
            vspan=vspan,
        )
        cc.add(
            tspan=(10, 11),       # dec
            period='month',
            vspan=vspan,
        )
        self._run_test(cc)

    def test_dayofyear_periods(self):
        vspan = (10, 20)
        cc = qartod.ClimatologyConfig()
        cc.add(
            tspan=(0, 1),       # first day of year
            period='dayofyear',
            vspan=vspan,
        )
        cc.add(
            tspan=(1, 363),       # jan 2 thru dec 30
            period='dayofyear',
            vspan=vspan,
        )
        cc.add(
            tspan=(363, 364),       # last day of year
            period='dayofyear',
            vspan=vspan,
        )
        self._run_test(cc)

    def test_weekofyear_periods(self):
        vspan = (10, 20)
        cc = qartod.ClimatologyConfig()
        cc.add(
            tspan=(0, 1),       # first week of year
            period='weekofyear',
            vspan=vspan,
        )
        cc.add(
            tspan=(1, 50),       # 2nd thru 51st week
            period='weekofyear',
            vspan=vspan,
        )
        cc.add(
            tspan=(50, 51),       # last week of year
            period='weekofyear',
            vspan=vspan,
        )
        self._run_test(cc)


class QartodClimatologyInclusiveRangesTest(unittest.TestCase):
    # Test that the various configuration spans (tspan, vspan, fspan, zspan) are
    # inclusive of both endpoints.
    def setUp(self):
        self.cc = qartod.ClimatologyConfig()
        self.cc.add(
            tspan=(np.datetime64('2019-11-01'), np.datetime64('2020-02-04')),
            fspan=(40, 70),
            vspan=(50, 60),
            zspan=(0, 10)
        )

    def _run_test(self, test_inputs, expected_result):
        times, values, depths = zip(*test_inputs)
        inputs = [
            values,
            np.asarray(values, dtype=np.float64),
            dask_arr(np.asarray(values, dtype=np.float64))
        ]

        for i in inputs:
            results = qartod.climatology_test(
                config=self.cc,
                tinp=times,
                inp=i,
                zinp=depths
            )
            npt.assert_array_equal(
                results,
                np.ma.array(expected_result)
            )

    def test_tspan_out_of_range_low(self):
        test_inputs = [
            (
                np.datetime64('2019-10-31'),
                55,
                5
            )
        ]
        expected_result = [2]
        self._run_test(test_inputs, expected_result)

    def test_tspan_minimum(self):
        test_inputs = [
            (
                np.datetime64('2019-11-01'),
                55,
                5
            )
        ]
        expected_result = [1]
        self._run_test(test_inputs, expected_result)

    def test_tspan_maximum(self):
        test_inputs = [
            (
                np.datetime64('2020-02-04'),
                55,
                5
            )
        ]
        expected_result = [1]
        self._run_test(test_inputs, expected_result)

    def test_tspan_out_of_range_high(self):
        test_inputs = [
            (
                np.datetime64('2020-02-05'),
                55,
                5
            )
        ]
        expected_result = [2]
        self._run_test(test_inputs, expected_result)

    def test_vspan_out_of_range_low(self):
        test_inputs = [
            (
                np.datetime64('2020-01-01'),
                49,
                5
            )
        ]
        expected_result = [3]
        self._run_test(test_inputs, expected_result)

    def test_vspan_minimum(self):
        test_inputs = [
            (
                np.datetime64('2020-01-01'),
                50,
                5
            )
        ]
        expected_result = [1]
        self._run_test(test_inputs, expected_result)

    def test_vspan_maximum(self):
        test_inputs = [
            (
                np.datetime64('2020-01-01'),
                60,
                5
            )
        ]
        expected_result = [1]
        self._run_test(test_inputs, expected_result)

    def test_vspan_out_of_range_high(self):
        test_inputs = [
            (
                np.datetime64('2020-01-01'),
                61,
                5
            )
        ]
        expected_result = [3]
        self._run_test(test_inputs, expected_result)

    def test_fspan_out_of_range_low(self):
        test_inputs = [
            (
                np.datetime64('2020-01-01'),
                30,
                5
            )
        ]
        expected_result = [4]
        self._run_test(test_inputs, expected_result)

    def test_fspan_minimum(self):
        test_inputs = [
            (
                np.datetime64('2020-01-01'),
                40,
                5
            )
        ]
        expected_result = [3]
        self._run_test(test_inputs, expected_result)

    def test_fspan_maximum(self):
        test_inputs = [
            (
                np.datetime64('2020-01-01'),
                70,
                5
            )
        ]
        expected_result = [3]
        self._run_test(test_inputs, expected_result)

    def test_fspan_out_of_range_high(self):
        test_inputs = [
            (
                np.datetime64('2020-01-01'),
                71,
                5
            )
        ]
        expected_result = [4]
        self._run_test(test_inputs, expected_result)

    def test_zspan_out_of_range_low(self):
        test_inputs = [
            (
                np.datetime64('2020-01-01'),
                55,
                -1
            )
        ]
        expected_result = [2]
        self._run_test(test_inputs, expected_result)

    def test_zspan_minimum(self):
        test_inputs = [
            (
                np.datetime64('2020-01-01'),
                55,
                0
            )
        ]
        expected_result = [1]
        self._run_test(test_inputs, expected_result)

    def test_zspan_maximum(self):
        test_inputs = [
            (
                np.datetime64('2020-01-01'),
                55,
                10
            )
        ]
        expected_result = [1]
        self._run_test(test_inputs, expected_result)

    def test_zspan_out_of_range_high(self):
        test_inputs = [
            (
                np.datetime64('2020-01-01'),
                55,
                11
            )
        ]
        expected_result = [2]
        self._run_test(test_inputs, expected_result)


class QartodClimatologyDepthTest(unittest.TestCase):

    def setUp(self):
        self.cc = qartod.ClimatologyConfig()
        # with depths
        self.cc.add(
            tspan=(np.datetime64('2012-01'), np.datetime64('2013-01')),
            vspan=(50, 60),
            zspan=(0, 10)
        )
        # same as above, but different depths
        self.cc.add(
            tspan=(np.datetime64('2012-01'), np.datetime64('2013-01')),
            vspan=(70, 80),
            zspan=(10, 100)
        )

    def _run_test(self, test_inputs, expected_result):
        times, values, depths = zip(*test_inputs)
        inputs = [
            values,
            np.asarray(values, dtype=np.float64),
            dask_arr(np.asarray(values, dtype=np.float64))
        ]

        for i in inputs:
            results = qartod.climatology_test(
                config=self.cc,
                tinp=times,
                inp=i,
                zinp=depths
            )
            npt.assert_array_equal(
                results,
                np.ma.array(expected_result)
            )

    def test_climatology_test_all_unknown(self):
        # Our configs only define depths, so this is never run if no
        # depths are passed in for any of the values
        test_inputs = [
            (
                np.datetime64('2011-01-02'),
                9,
                None
            ),
            (
                np.datetime64('2011-01-02'),
                11,
                None
            ),
            (
                np.datetime64('2011-01-02'),
                21,
                None
            ),
            # not run, outside given time ranges
            (
                np.datetime64('2015-01-02'),
                21,
                None
            ),
        ]
        expected_result = [2, 2, 2, 2]
        self._run_test(test_inputs, expected_result)


class QartodClimatologyTest(unittest.TestCase):

    def setUp(self):
        self.cc = qartod.ClimatologyConfig()
        self.cc.add(
            tspan=(np.datetime64('2011-01'), np.datetime64('2011-07')),
            vspan=(10, 20)
        )
        self.cc.add(
            tspan=(np.datetime64('2011-07'), np.datetime64('2012-01')),
            vspan=(30, 40)
        )
        self.cc.add(
            tspan=(np.datetime64('2012-01'), np.datetime64('2013-01')),
            vspan=(40, 50),
        )
        # same time range as above, but with depths
        self.cc.add(
            tspan=(np.datetime64('2012-01'), np.datetime64('2013-01')),
            vspan=(50, 60),
            zspan=(0, 10)
        )
        # same as above, but different depths
        self.cc.add(
            tspan=(np.datetime64('2012-01'), np.datetime64('2013-01')),
            vspan=(70, 80),
            zspan=(10, 100)
        )

    def _run_test(self, test_inputs, expected_result):
        times, values, depths = zip(*test_inputs)
        inputs = [
            values,
            np.asarray(values, dtype=np.float64),
            dask_arr(np.asarray(values, dtype=np.float64))
        ]

        for i in inputs:
            results = qartod.climatology_test(
                config=self.cc,
                tinp=times,
                inp=i,
                zinp=depths
            )
            npt.assert_array_equal(
                results,
                np.ma.array(expected_result)
            )

    def test_climatology_test(self):
        test_inputs = [
            (
                np.datetime64('2011-01-02'),
                11,
                None
            )
        ]
        expected_result = [1]
        self._run_test(test_inputs, expected_result)

    def test_climatology_test_seconds_since_epoch(self):
        test_inputs = [
            (
                1293926400,  # Sunday, January 2, 2011 12:00:00 AM UTC
                11,
                None
            )
        ]
        expected_result = [1]
        self._run_test(test_inputs, expected_result)

    def test_climatology_test_fail(self):
        test_inputs = [
            (
                np.datetime64('2011-01-02'),
                9,
                None
            ),
            (
                np.datetime64('2011-01-02'),
                11,
                None
            ),
            (
                np.datetime64('2011-01-02'),
                21,
                None
            ),
            # not run, outside given time ranges
            (
                np.datetime64('2015-01-02'),
                21,
                None
            ),
        ]
        expected_result = [3, 1, 3, 2]
        self._run_test(test_inputs, expected_result)

    def test_climatology_test_depths(self):
        test_inputs = [
            # (0, 10) depth range, valid value
            (
                np.datetime64('2012-01-02'),
                51,
                2
            ),
            # (10, 100) depth range, valid value
            (
                np.datetime64('2012-01-02'),
                71,
                90
            ),
            # no depth range, valid value
            (
                np.datetime64('2012-01-02'),
                42,
                None
            ),
            # no depth range, invalid value
            (
                np.datetime64('2012-01-02'),
                39,
                None
            ),
            # (10, 100) depth range, invalid value
            (
                np.datetime64('2012-01-02'),
                59,
                11
            ),
            # Not run, has depth that's outside of given depth ranges
            (
                np.datetime64('2012-01-02'),
                79,
                101
            )
        ]
        expected_result = [1, 1, 1, 3, 3, 2]
        self._run_test(test_inputs, expected_result)


class QartodSpikeTest(unittest.TestCase):

    def setUp(self):
        self.suspect_threshold = 25
        self.fail_threshold = 50

    def test_spike(self):
        """
        Test to make ensure single value spike detection works properly.
        """

        arr = [10, 12, 999.99, 13, 15, 40, 9, 9]

        # First and last elements should always be good data, unless someone
        # has set a threshold to zero.
        expected = [2, 4, 4, 4, 1, 3, 1, 2]

        inputs = [
            arr,
            np.asarray(arr, dtype=np.float64),
            dask_arr(np.asarray(arr, dtype=np.float64))
        ]
        for i in inputs:
            npt.assert_array_equal(
                qartod.spike_test(
                    inp=i,
                    suspect_threshold=self.suspect_threshold,
                    fail_threshold=self.fail_threshold
                ),
                expected
            )

    def test_spike_negative_vals(self):
        """
        Test to make spike detection works properly for negative values.
        """
        arr = [-10, -12, -999.99, -13, -15, -40, -9, -9]

        # First and last elements should always be good data, unless someone
        # has set a threshold to zero.
        expected = [2, 4, 4, 4, 1, 3, 1, 2]

        inputs = [
            arr,
            np.asarray(arr, dtype=np.float64),
            dask_arr(np.asarray(arr, dtype=np.float64))
        ]
        for i in inputs:
            npt.assert_array_equal(
                qartod.spike_test(
                    inp=i,
                    suspect_threshold=self.suspect_threshold,
                    fail_threshold=self.fail_threshold
                ),
                expected
            )

    def test_spike_initial_final_values(self):
        """
        The test is not defined for the initial and final values in the array
        """
        arr = [-100, -99, -99, -98]
        expected = [2, 1, 1, 2]

        npt.assert_array_equal(
            qartod.spike_test(
                inp=arr,
                suspect_threshold=self.suspect_threshold,
                fail_threshold=self.fail_threshold
            ),
            expected
        )

    def test_spike_masked(self):
        """
        Test with missing data.
        """

        arr = [10, 12, 999.99, 13, 15, 40, 9, 9, None, 10, 10, 999.99, 10, None]

        # First and last elements should always be good data, unless someone
        # has set a threshold to zero.
        expected = [2, 4, 4, 4, 1, 3, 1, 1, 9, 9, 4, 4, 4, 9]

        inputs = [
            arr,
            np.asarray(arr, dtype=np.float64),
            dask_arr(np.asarray(arr, dtype=np.float64))
        ]
        for i in inputs:
            npt.assert_array_equal(
                qartod.spike_test(
                    inp=i,
                    suspect_threshold=self.suspect_threshold,
                    fail_threshold=self.fail_threshold
                ),
                expected
            )

    def test_spike_realdata(self):
        """
        Test with real-world data.
        """
        suspect_threshold = 0.5
        fail_threshold = 1

        arr = [-0.189, -0.0792, -0.0122, 0.0457, 0.0671, 0.0213, -0.0488, -0.1463, -0.2438, -0.3261, -0.3871, -0.4054,
               -0.3932, -0.3383, -0.2804, -0.2347, -0.2134, -0.2347, -0.2926, -0.3597, -0.442, -0.509, 0, -0.5944,
               -0.57, -0.4267, -0.2926, -0.1585, -0.0945, -0.0762]

        expected = [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 2]

        inputs = [
            arr,
            np.asarray(arr, dtype=np.float64),
            dask_arr(np.asarray(arr, dtype=np.float64))
        ]
        for i in inputs:
            npt.assert_array_equal(
                qartod.spike_test(
                    inp=i,
                    suspect_threshold=suspect_threshold,
                    fail_threshold=fail_threshold
                ),
                expected
            )


    def test_spike_methods(self):
        """
        Test the different input methods and review the different flags expected.
        """
        inp = [3, 4.99, 5, 6, 8, 6, 6, 6.75, 6, 6, 5.3, 6, 6, 9, 5, None, 4, 4]
        suspect_threshold = .5
        fail_threshold = 1
        average_method_expected = [2, 3, 1, 1, 4, 3, 1, 3, 1, 1, 3, 1, 4, 4, 4, 9, 9, 2]
        diff_method_expected = [2, 1, 1, 1, 4, 1, 1, 3, 1, 1, 3, 1, 1, 4, 9, 9, 9, 2]

        # Test average method
        npt.assert_array_equal(
            qartod.spike_test(
                inp=inp,
                suspect_threshold=suspect_threshold,
                fail_threshold=fail_threshold,
                method='average'
            ),
            average_method_expected
        )

        # Test diff method
        npt.assert_array_equal(
            qartod.spike_test(
                inp=inp,
                suspect_threshold=suspect_threshold,
                fail_threshold=fail_threshold,
                method='differential'
            ),
            diff_method_expected
        )

        # Test default method (average)
        npt.assert_array_equal(
            qartod.spike_test(
                inp=inp,
                suspect_threshold=suspect_threshold,
                fail_threshold=fail_threshold,
            ),
            average_method_expected
        )


    def test_spike_test_bad_method(self):
        inp = [3, 4.99, 5, 6, 8, 6, 6, 6.75, 6, 6, 5.3, 6, 6, 9, 5, None, 4, 4]
        suspect_threshold = .5
        fail_threshold = 1

        with self.assertRaises(ValueError):
            qartod.spike_test(
                inp=inp,
                suspect_threshold=suspect_threshold,
                fail_threshold=fail_threshold,
                method='bad'
            )
        with self.assertRaises(ValueError):
            qartod.spike_test(
                inp=inp,
                suspect_threshold=suspect_threshold,
                fail_threshold=fail_threshold,
                method=123
            )


    def test_spike_test_inputs(self):
        inp = [3, 4.99, 5, 6, 8, 6, 6, 6.75, 6, 6, 5.3, 6, 6, 9, 5, None, 4, 4]
        expected_suspect_only = [2, 3, 1, 1, 3, 3, 1, 3, 1, 1, 3, 1, 3, 3, 3, 9, 9, 2]
        expected_fail_only = [2, 1, 1, 1, 4, 1, 1, 1, 1, 1, 1, 1, 4, 4, 4, 9, 9, 2]
        suspect_threshold = .5
        fail_threshold = 1

        npt.assert_array_equal(
            qartod.spike_test(
                inp=inp,
                suspect_threshold=suspect_threshold,
            ),
            expected_suspect_only)
        npt.assert_array_equal(
            qartod.spike_test(
                inp=inp,
                fail_threshold=fail_threshold,
            ),
            expected_fail_only)


class QartodRateOfChangeTest(unittest.TestCase):

    def setUp(self):
        self.times = np.arange('2015-01-01 00:00:00', '2015-01-01 06:00:00',
                               step=np.timedelta64(15, 'm'), dtype=np.datetime64)
        self.times_epoch_secs = [t.astype(int) for t in self.times]
        self.threshold = 5 / 15 / 60  # 5 units per 15 minutes --> 5/15/60 units per second

    def test_rate_of_change(self):
        times = self.times
        arr = [2, 10, 2.1, 3, 4, 5, 7, 10, 0, 2, 2.2, 2, 1, 2, 3, 90, 91, 92, 93, 1, 2, 3, 4, 5]
        expected = [1, 3, 3, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 3, 1, 1, 1, 1]
        inputs = [
            arr,
            np.asarray(arr, dtype=np.float64),
            dask_arr(np.asarray(arr, dtype=np.float64))
        ]
        for i in inputs:
            result = qartod.rate_of_change_test(
                inp=i,
                tinp=times,
                threshold=self.threshold
            )
            npt.assert_array_equal(expected, result)

        # test epoch secs - should return same result
        npt.assert_array_equal(
            qartod.rate_of_change_test(
                inp=i,
                tinp=self.times_epoch_secs,
                threshold=self.threshold
            ),
            expected
        )

    def test_rate_of_change_missing_values(self):
        times = self.times[0:8]
        arr = [2, 10, 2, 3, None, None, 7, 10]
        expected = [1, 3, 3, 1, 9, 9, 1, 1]
        result = qartod.rate_of_change_test(
            inp=arr,
            tinp=times,
            threshold=self.threshold
        )
        npt.assert_array_equal(expected, result)

    def test_rate_of_change_negative_values(self):
        times = self.times[0:4]
        arr = [-2, -10, -2, -3]
        expected = [1, 3, 3, 1]
        result = qartod.rate_of_change_test(
            inp=arr,
            tinp=times,
            threshold=self.threshold
        )
        npt.assert_array_equal(expected, result)


class QartodFlatLineTest(unittest.TestCase):

    def setUp(self):
        self.times = np.arange('2015-01-01 00:00:00', '2015-01-01 03:30:00',
                               step=np.timedelta64(15, 'm'), dtype=np.datetime64)
        self.times_epoch_secs = [t.astype(int) for t in self.times]
        self.suspect_threshold = 3000   # 50 mins, or count of 3
        self.fail_threshold = 4800  # 80 mins, or count of 5
        self.tolerance = 0.01

    def test_flat_line(self):
        arr = [1, 2, 2.0001, 2, 2.0001, 2, 2.0001, 2, 4, 5, 3, 3.0001, 3.0005, 3.00001]
        expected = [1, 1, 1, 1, 3, 3, 4, 4, 1, 1, 1, 1, 1, 3]
        inputs = [
            arr,
            np.asarray(arr, dtype=np.float64),
            dask_arr(np.asarray(arr, dtype=np.float64))
        ]
        for i in inputs:
            result = qartod.flat_line_test(
                inp=i,
                tinp=self.times,
                suspect_threshold=self.suspect_threshold,
                fail_threshold=self.fail_threshold,
                tolerance=self.tolerance
            )
            npt.assert_array_equal(result, expected)

        # test epoch secs - should return same result
        npt.assert_array_equal(
            qartod.flat_line_test(
                inp=arr,
                tinp=self.times_epoch_secs,
                suspect_threshold=self.suspect_threshold,
                fail_threshold=self.fail_threshold,
                tolerance=self.tolerance
            ),
            expected
        )

        # test negative array - should return same result
        arr = [-1 * x for x in arr]
        npt.assert_array_equal(
            qartod.flat_line_test(
                inp=arr,
                tinp=self.times,
                suspect_threshold=self.suspect_threshold,
                fail_threshold=self.fail_threshold,
                tolerance=self.tolerance
            ),
            expected
        )

        # test empty array - should return empty result
        arr = np.array([])
        expected = np.array([])
        npt.assert_array_equal(
            qartod.flat_line_test(
                inp=arr,
                tinp=self.times,
                suspect_threshold=self.suspect_threshold,
                fail_threshold=self.fail_threshold,
                tolerance=self.tolerance
            ),
            expected
        )

        # test nothing fails
        arr = np.random.random(len(self.times))
        expected = np.ones_like(arr)
        npt.assert_array_equal(
            qartod.flat_line_test(
                inp=arr,
                tinp=self.times,
                suspect_threshold=self.suspect_threshold,
                fail_threshold=self.fail_threshold,
                tolerance=0.00000000001
            ),
            expected
        )

    def test_flat_line_starting_from_beginning(self):
        arr = [2, 2.0001, 2, 2.0001, 2, 2.0001, 2, 4, 5, 3, 3.0001, 3.0005, 3.00001]
        expected = [1, 1, 1, 3, 3, 4, 4, 1, 1, 1, 1, 1, 3]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            inputs = [
                arr,
                np.asarray(arr, dtype=np.float64),
                dask_arr(np.asarray(arr, dtype=np.float64))
            ]
        for i in inputs:
            result = qartod.flat_line_test(
                inp=i,
                tinp=self.times,
                suspect_threshold=self.suspect_threshold,
                fail_threshold=self.fail_threshold,
                tolerance=self.tolerance
            )
            npt.assert_array_equal(result, expected)

    def test_flat_line_short_timeseries(self):

        def check(time, arr, expected):
            result = qartod.flat_line_test(
                inp=arr,
                tinp=time,
                suspect_threshold=3,
                fail_threshold=5,
                tolerance=0.1
            )
            npt.assert_array_equal(result, expected)

        check(time=[],                  arr=[],                     expected=[])
        check(time=[0],                 arr=[5],                    expected=[1])
        check(time=[0, 1],              arr=[5, 5],                 expected=[1, 1])
        check(time=[0, 1, 2],           arr=[5, 5, 5],              expected=[1, 1, 1])
        check(time=[0, 1, 2, 3],        arr=[5, 5, 5, 5],           expected=[1, 1, 1, 3])
        check(time=[0, 1, 2, 3, 4],     arr=[5, 5, 5, 5, 5],        expected=[1, 1, 1, 3, 3])
        check(time=[0, 1, 2, 3, 4, 5],  arr=[5, 5, 5, 5, 5, 5],     expected=[1, 1, 1, 3, 3, 4])

    def test_flat_line_with_spike(self):
        tolerance = 4
        suspect_threshold = 3
        fail_threshold = 6
        time = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        arr = [1, 1, 1, 1, 1, 6, 5, 4, 3, 2]
        expected = [1, 1, 1, 3, 3, 1, 1, 1, 3, 3]
        result = qartod.flat_line_test(
            inp=arr,
            tinp=time,
            suspect_threshold=suspect_threshold,
            fail_threshold=fail_threshold,
            tolerance=tolerance
        )
        npt.assert_array_equal(result, expected)

    def test_flat_line_missing_values(self):
        arr = [1, None, np.ma.masked, 2, 2.0001, 2, 2.0001, 2, 4, None, 3, None, None, 3.00001]
        expected = [1, 9, 9, 1, 3, 3, 4, 4, 1, 9, 1, 9, 9, 3]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            inputs = [
                arr,
                np.asarray(arr, dtype=np.float64),
                dask_arr(np.asarray(arr, dtype=np.float64))
            ]
        for i in inputs:
            result = qartod.flat_line_test(
                inp=i,
                tinp=self.times,
                suspect_threshold=self.suspect_threshold,
                fail_threshold=self.fail_threshold,
                tolerance=self.tolerance
            )
            npt.assert_array_equal(result, expected)


class QartodAttenuatedSignalTest(unittest.TestCase):

    def _run_test(self, times, signal, suspect_threshold, fail_threshold, check_type, expected,
                  test_period=None, min_obs=None, min_period=None):
        npt.assert_array_equal(
            qartod.attenuated_signal_test(
                inp=signal,
                tinp=times,
                suspect_threshold=suspect_threshold,
                fail_threshold=fail_threshold,
                test_period=test_period,
                min_obs=min_obs,
                min_period=min_period,
                check_type=check_type
            ),
            expected
        )

        # test epoch secs - should return same result
        times_epoch_secs = [np.datetime64(t, 's').astype(int) for t in times]
        npt.assert_array_equal(
            qartod.attenuated_signal_test(
                inp=signal,
                tinp=times_epoch_secs,
                suspect_threshold=suspect_threshold,
                fail_threshold=fail_threshold,
                test_period=test_period,
                min_obs=min_obs,
                min_period=min_period,
                check_type=check_type
            ),
            expected
        )

    def test_attenuated_signal(self):
        # good signal, all pass
        signal = np.array([1, 2, 3, 4])
        times = np.array([
            np.datetime64('2019-01-01') + np.timedelta64(i, 'D') for i in range(signal.size)
        ])
        expected = np.array([1, 1, 1, 1])
        self._run_test(times=times, signal=signal,
                       suspect_threshold=0.75, fail_threshold=0.5, check_type='std',
                       expected=expected)

        # Only suspect
        signal = np.array([1, 2, 3, 4])
        times = np.array([
            np.datetime64('2019-01-01') + np.timedelta64(i, 'D') for i in range(signal.size)
        ])
        expected = np.array([3, 3, 3, 3])
        self._run_test(times=times, signal=signal,
                       suspect_threshold=5, fail_threshold=0, check_type='std',
                       expected=expected)

        # Not changing should fail
        signal = np.array([1, 1, 1, 1])
        times = np.array([
            np.datetime64('2019-01-01') + np.timedelta64(i, 'D') for i in range(signal.size)
        ])
        expected = np.array([4, 4, 4, 4])
        self._run_test(times=times, signal=signal,
                       suspect_threshold=10, fail_threshold=8, check_type='std',
                       expected=expected)

        # std deviation less than fail threshold
        signal = np.array([10, 20, 30, 40])
        times = np.array([
            np.datetime64('2019-01-01') + np.timedelta64(i, 'D') for i in range(signal.size)
        ])
        expected = np.array([4, 4, 4, 4])
        self._run_test(times=times, signal=signal,
                       suspect_threshold=100000, fail_threshold=1000, check_type='std',
                       expected=expected)

    def test_attenuated_signal_range(self):
        # range less than fail threshold
        signal = np.array([10, 20, 30, 40])
        times = np.array([
            np.datetime64('2019-01-01') + np.timedelta64(i, 'D') for i in range(signal.size)
        ])
        expected = np.array([4, 4, 4, 4])
        self._run_test(times=times, signal=signal,
                       suspect_threshold=50, fail_threshold=31, check_type='range',
                       expected=expected)

        # range less than suspect threshold
        signal = np.array([10, 20, 30, 40])
        times = np.array([
            np.datetime64('2019-01-01') + np.timedelta64(i, 'D') for i in range(signal.size)
        ])
        expected = np.array([3, 3, 3, 3])
        self._run_test(times=times, signal=signal,
                       suspect_threshold=31, fail_threshold=10, check_type='range',
                       expected=expected)

        signal = np.array([3, 4, 5, 8.1, 9, 8.5, 8.7, 8.4, 8.2, 8.35, 2, 1])
        times = np.array([
            np.datetime64('2019-01-01') + np.timedelta64(i, 'D') for i in range(signal.size)
        ])
        expected = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        self._run_test(times=times, signal=signal,
                       suspect_threshold=0.15, fail_threshold=0.1, check_type='range',
                       expected=expected)

    def test_attenuated_signal_time_window(self):
        # test time windowed range
        signal = [1, 2, 3, 100, 1000]
        times = np.array([
            np.datetime64('2019-01-01') + np.timedelta64(i, 'D') for i in range(len(signal))
        ])
        time_window = 2 * 86400     # 2 days

        def _run_test_time_window(min_obs, min_period, expected):
            self._run_test(times=times, signal=signal,
                           suspect_threshold=100, fail_threshold=10, check_type='range',
                           expected=expected,
                           test_period=time_window,
                           min_obs=min_obs,
                           min_period=min_period)

        # zero min_obs -- initial values should fail
        min_obs = 0
        min_period = None
        expected = [4, 4, 4, 3, 1]
        _run_test_time_window(min_obs, min_period, expected)

        # zero min_period -- initial values should fail
        min_obs = None
        min_period = 0
        expected = [4, 4, 4, 3, 1]
        _run_test_time_window(min_obs, min_period, expected)

        # min_obs the same size as time_window -- first window should be UNKNOWN
        min_obs = 2     # 2 days (since 1 obs per day)
        min_period = None
        expected = [2, 4, 4, 3, 1]
        _run_test_time_window(min_obs, min_period, expected)

        # min_period the same size as time_window -- first window should be UNKNOWN
        min_obs = None
        min_period = time_window
        expected = [2, 4, 4, 3, 1]
        _run_test_time_window(min_obs, min_period, expected)

    def test_attenuated_signal_missing(self):
        signal = np.array([None, 2, 3, 4])
        times = np.array([
            np.datetime64('2019-01-01') + np.timedelta64(i, 'D') for i in range(signal.size)
        ])
        expected = np.array([9, 1, 1, 1])
        self._run_test(times=times, signal=signal,
                       suspect_threshold=0.75, fail_threshold=0.5, check_type='std',
                       expected=expected)

        signal = np.array([None, None, None, None])
        times = np.array([
            np.datetime64('2019-01-01') + np.timedelta64(i, 'D') for i in range(signal.size)
        ])
        expected = np.array([9, 9, 9, 9])
        self._run_test(times=times, signal=signal,
                       suspect_threshold=0.75, fail_threshold=0.5, check_type='std',
                       expected=expected)

        # range less than 30
        signal = [10, None, None, 40]
        times = np.array([
            np.datetime64('2019-01-01') + np.timedelta64(i, 'D') for i in range(len(signal))
        ])
        expected = np.array([4, 9, 9, 4])
        self._run_test(times=times, signal=signal,
                       suspect_threshold=50, fail_threshold=31, check_type='range',
                       expected=expected)

    def test_attenuated_signal_missing_time_window(self):
        # test time windowed range with missing values
        signal = [1, None, 10, 100, 1000]
        times = np.array([
            np.datetime64('2019-01-01') + np.timedelta64(i, 'D') for i in range(len(signal))
        ])
        time_window = 2 * 86400     # 2 days
        min_obs = 2                 # 2 days (since 1 obs per day)

        # test time windowed range
        expected = [2, 9, 2, 3, 1]
        self._run_test(times=times, signal=signal,
                       suspect_threshold=100, fail_threshold=50, check_type='range',
                       expected=expected,
                       test_period=time_window,
                       min_obs=min_obs)

        # test time windowed std
        expected = [2, 9, 2, 3, 1]
        time_window = 2 * 86400
        self._run_test(times=times, signal=signal,
                       suspect_threshold=150, fail_threshold=40, check_type='std',
                       expected=expected,
                       test_period=time_window,
                       min_obs=min_obs)


class QartodDensityInversionTest(unittest.TestCase):

    def _run_density_inversion_tests(self, density, depth, result,
                             suspect_threshold=-0.01,
                             fail_threshold=-.03):
        # Try every possible input format combinations
        dens_inputs = [
            density,
            np.asarray(density, dtype=np.float64),
            dask_arr(np.asarray(density, dtype=np.float64))]

        depth_inputs = [
            depth,
            np.asarray(depth, dtype=np.float64),
            dask_arr(np.asarray(depth, dtype=np.float64))]

        for rho in dens_inputs:
            for z in depth_inputs:
                npt.assert_array_equal(qartod.density_inversion_test(inp=rho, zinp=z,
                                                                     suspect_threshold=suspect_threshold,
                                                                     fail_threshold=fail_threshold),
                                       result)

    def test_density_inversion_downcast_flags(self):
        depth = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        density = [1024, 1024, 1023.98, 1024, 1025, 1026, 1025.9, 1026, 1026, None, 1026, 1027]
        result = [1, 3, 3, 1, 1, 4, 4, 1, 1, 9, 9, 1]
        self._run_density_inversion_tests(density, depth, result)

    def test_density_inversion_upcast_flags(self):
        depth = [11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
        density = [1026, None, 1026, 1026, 1025.9, 1026, 1025, 1024, 1023.98, 1024, 1024]
        result = [1, 9, 9, 1, 4, 4, 1, 1, 3, 3, 1]
        self._run_density_inversion_tests(density, depth, result)

    def test_density_inversion_down_up_cast_flags(self):
        depth = [1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 8, 7, 6, 5, 4, 3, 2, 1]
        density = [1024, 1024, 1023.98, 1024, 1025, 1026, 1025.9, 1026, 1026,
                   1026, 1026, 1025.9, 1026, 1025, 1024, 1023.98, 1024, 1024]
        result = [1, 3, 3, 1, 1, 4, 4, 1, 1, 1, 1, 4, 4, 1, 1, 3, 3, 1]
        self._run_density_inversion_tests(density, depth, result)

    def test_density_inversion_stable_depth_flags(self):
        depth = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        density = [1026, None, 1026, 1026, 1025.9, 1026, 1025, 1024, 1023.98, 1024, 1024]
        result = [1, 9, 9, 1, 1, 1, 1, 1, 1, 1, 1]
        self._run_density_inversion_tests(density, depth, result)

    def test_density_inversion_one_record_input(self):
        # One Value test
        depth = [1]
        density = [1026]
        result = [2]
        self._run_density_inversion_tests(density, depth, result)

    def test_density_inversion_bad_depth_value(self):
        # Missing depth value
        depth = [1, None, 3, 4, 5]
        density = [1025, 1025, 1025, 1026, 1026]
        result = [1, 9, 9, 1, 1]
        self._run_density_inversion_tests(density, depth, result)

    def test_density_inversion_input(self):
        density = [1024, 1024, 1025]
        depth = [1, 2, 3]

        # Wrong type suspect_threshold
        with self.assertRaises(TypeError):
            qartod.density_inversion_test(inp=density, zinp=depth, suspect_threshold='bad')

        # Wrong type fail_threshold
        with self.assertRaises(TypeError):
            qartod.density_inversion_test(inp=density, zinp=depth, fail_threshold='bad')

        # Wrong type for both fail_threshold and suspect_threshold
        with self.assertRaises(TypeError):
            qartod.density_inversion_test(inp=density, zinp=depth,
                                          suspect_threshold='bad', fail_threshold='bad')

        # Wrong type density
        with self.assertRaises(ValueError):
            qartod.density_inversion_test(inp='density', zinp=depth, suspect_threshold=-0.3)

        # Wrong type depth
        with self.assertRaises(ValueError):
            qartod.density_inversion_test(inp=density, zinp='depth', suspect_threshold=-0.3)


class QartodUtilsTests(unittest.TestCase):

    def test_qartod_compare(self):
        """
        Tests that the compare function works as intended.
        """

        range_flags = np.array([1, 1, 1, 9, 1, 1, 9, 9])
        spike_flags = np.array([2, 1, 1, 1, 1, 1, 9, 9])
        grdtn_flags = np.array([1, 3, 3, 4, 3, 1, 2, 9])

        primary_flags = qartod.qartod_compare([
            range_flags,
            spike_flags,
            grdtn_flags
        ])
        np.testing.assert_array_equal(
            primary_flags,
            np.array([1, 3, 3, 4, 3, 1, 2, 9])
        )
