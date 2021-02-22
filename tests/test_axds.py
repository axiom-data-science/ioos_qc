#!/usr/bin/env python
# coding=utf-8
import logging
import unittest
from datetime import datetime
from functools import partial

import numpy as np
import numpy.testing as npt

import ioos_qc
from ioos_qc import axds
from ioos_qc.streams import NumpyStream
from ioos_qc.results import collect_results
from ioos_qc.config import Config, Context, Call, tw

L = logging.getLogger('ioos_qc')
L.setLevel(logging.INFO)
L.addHandler(logging.StreamHandler())


class AxdsValidTimeBoundsTest(unittest.TestCase):

    def setUp(self):
        self.times = np.arange('2015-01-01 00:00:00', '2015-01-01 06:00:00',
                               step=np.timedelta64(1, 'h'), dtype=np.datetime64)

    def test_no_bounds(self):
        valid_spans = [
            (
                np.datetime64("NaT"),
                np.datetime64("NaT")
            ),
            (
                None,
                None
            )
        ]

        for valid_span in valid_spans:
            npt.assert_array_equal(
                axds.valid_range_test(self.times,
                                      valid_span=valid_span),
                np.array([1, 1, 1, 1, 1, 1])
            )

    def test_chop_start(self):
        valid_spans = [
            (
                np.datetime64('2015-01-01T02:00:00'),
                np.datetime64("NaT"),
            ),
            (
                np.datetime64('2015-01-01T02:00:00').astype(datetime),
                None
            )
        ]

        for valid_span in valid_spans:
            npt.assert_array_equal(
                axds.valid_range_test(self.times,
                                      valid_span=valid_span),
                np.array([4, 4, 1, 1, 1, 1])
            )

    def test_chop_end(self):
        valid_spans = [
            (
                np.datetime64("NaT"),
                np.datetime64('2015-01-01T04:00:00'),
            ),
            (
                None,
                np.datetime64('2015-01-01T04:00:00').astype(datetime),
            )
        ]

        for valid_span in valid_spans:
            npt.assert_array_equal(
                axds.valid_range_test(self.times,
                                      valid_span=valid_span),
                np.array([1, 1, 1, 1, 4, 4])
            )

    def test_chop_ends(self):
        valid_spans = [
            (
                np.datetime64('2015-01-01T02:00:00'),
                np.datetime64('2015-01-01T04:00:00')
            ),
            (
                np.datetime64('2015-01-01T02:00:00').astype(datetime),
                np.datetime64('2015-01-01T04:00:00').astype(datetime)
            )
        ]

        for valid_span in valid_spans:
            npt.assert_array_equal(
                axds.valid_range_test(self.times,
                                      valid_span=valid_span),
                np.array([4, 4, 1, 1, 4, 4])
            )

    def test_chop_all(self):
        valid_spans = [
            (
                np.datetime64('2000-12-01T00:00:00'),
                np.datetime64('2001-12-01T00:00:00')
            ),
            (
                np.datetime64('2000-12-01T00:00:00').astype(datetime),
                np.datetime64('2001-12-01T00:00:00').astype(datetime)
            )
        ]

        for valid_span in valid_spans:
            npt.assert_array_equal(
                axds.valid_range_test(self.times,
                                      valid_span=valid_span),
                np.array([4, 4, 4, 4, 4, 4])
            )

    def test_empty_chop_ends(self):
        valid_span = (
            np.datetime64("NaT"),
            np.datetime64('2015-01-01T04:00:00'),
        )

        times = np.arange('2015-01-01 00:00:00', '2015-01-01 06:00:00',
                           step=np.timedelta64(1, 'h'), dtype=np.datetime64)
        times[0:2] = np.datetime64("NaT")

        npt.assert_array_equal(
            axds.valid_range_test(times,
                                  valid_span=valid_span),
            np.array([9, 9, 1, 1, 4, 4])
        )

    def test_all_empty(self):
        valid_span = (
            np.datetime64("NaT"),
            np.datetime64("NaT")
        )

        times = np.arange('2015-01-01 00:00:00', '2015-01-01 06:00:00',
                          step=np.timedelta64(1, 'h'), dtype=np.datetime64)
        times[:] = np.datetime64("NaT")

        npt.assert_array_equal(
            axds.valid_range_test(times,
                                  valid_span=valid_span),
            np.array([9, 9, 9, 9, 9, 9])
        )

    def test_inclusive_exclusive(self):
        valid_spans = [
            (
                np.datetime64('2015-01-01T02:00:00'),
                np.datetime64('2015-01-01T04:00:00')
            ),
            (
                np.datetime64('2015-01-01T02:00:00').astype(datetime),
                np.datetime64('2015-01-01T04:00:00').astype(datetime)
            )
        ]

        for valid_span in valid_spans:
            npt.assert_array_equal(
                axds.valid_range_test(self.times,
                                      valid_span=valid_span),
                np.array([4, 4, 1, 1, 4, 4])
            )

        for valid_span in valid_spans:
            npt.assert_array_equal(
                axds.valid_range_test(self.times,
                                      valid_span=valid_span,
                                      start_inclusive=True,
                                      end_inclusive=False),
                np.array([4, 4, 1, 1, 4, 4])
            )

        for valid_span in valid_spans:
            npt.assert_array_equal(
                axds.valid_range_test(self.times,
                                      valid_span=valid_span,
                                      start_inclusive=True,
                                      end_inclusive=True),
                np.array([4, 4, 1, 1, 1, 4])
            )

        for valid_span in valid_spans:
            npt.assert_array_equal(
                axds.valid_range_test(self.times,
                                      valid_span=valid_span,
                                      start_inclusive=False,
                                      end_inclusive=True),
                np.array([4, 4, 4, 1, 1, 4])
            )

        for valid_span in valid_spans:
            npt.assert_array_equal(
                axds.valid_range_test(self.times,
                                      valid_span=valid_span,
                                      start_inclusive=False,
                                      end_inclusive=False),
                np.array([4, 4, 4, 1, 4, 4])
            )

    def test_with_config(self):

        config_str = """
            variable1:
                axds:
                    valid_range_test:
                        valid_span: ['2015-01-01T02:00:00', '2015-01-01T04:00:00']
        """

        config = Config(config_str)

        assert config.calls == [
            Call(
                stream_id='variable1',
                context=Context(),
                call=partial(
                    ioos_qc.axds.valid_range_test,
                    (),
                    valid_span=['2015-01-01T02:00:00', '2015-01-01T04:00:00']
                )
            )
        ]

        ns = NumpyStream(self.times)
        results = ns.run(config)
        results = collect_results(results, how='dict')