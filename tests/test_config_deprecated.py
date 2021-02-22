#!/usr/bin/env python
# coding=utf-8
import io
import os
import logging
import tempfile
import unittest
import simplejson as json
from copy import deepcopy
from pathlib import Path

import numpy as np
import netCDF4 as nc4
import numpy.testing as npt
from ruamel.yaml import YAML

from ioos_qc.utils import GeoNumpyDateEncoder
from ioos_qc.config import QcConfig, NcQcConfig
from ioos_qc.qartod import ClimatologyConfig


L = logging.getLogger('ioos_qc')
L.setLevel(logging.INFO)
L.handlers = [logging.StreamHandler()]

yaml = YAML(typ='safe')


class ConfigLoadTest(unittest.TestCase):

    def setUp(self):
        template = """
        qartod:
            gross_range_test:
                suspect_span: [1, 11]
                fail_span:
                    - 0
                    - 12
            goober:
                foo: [1, null]
        """
        self.handle, self.yamlfile = tempfile.mkstemp(suffix='.yaml')
        with open(self.yamlfile, 'w') as f:
            f.write(template)

        self.expected_dict = {
            'qartod': {
                'gross_range_test': {
                    'suspect_span': [1, 11],
                    'fail_span': [0, 12],
                },
                'goober': {
                    'foo': [1, None]
                }
            }
        }

    def tearDown(self):
        os.close(self.handle)
        os.remove(self.yamlfile)

    def test_load_yaml_dict_object(self):
        with open(self.yamlfile) as f:
            y = yaml.load(f.read())
            qc = QcConfig(y)
        assert qc.config == self.expected_dict

    def test_load_yaml_str(self):
        with open(self.yamlfile) as f:
            qc = QcConfig(f.read())
        assert qc.config == self.expected_dict

    def test_load_json_str(self):
        with open(self.yamlfile) as f:
            js = json.dumps(yaml.load(f.read()))
        qc = QcConfig(js)
        assert qc.config == self.expected_dict

    def test_load_yaml_file_path(self):
        qc = QcConfig(self.yamlfile)
        assert qc.config == self.expected_dict

    def test_load_yaml_path_object(self):
        qc = QcConfig(Path(self.yamlfile))
        assert qc.config == self.expected_dict

    def test_load_json_stringio(self):
        st = io.StringIO()
        qc = QcConfig(self.yamlfile)
        with open(self.yamlfile, 'rt') as f:
            js = json.dumps(yaml.load(f.read()))
            st.write(js)
        qc = QcConfig(st)
        st.close()
        assert qc.config == self.expected_dict

    def test_load_yaml_stringio(self):
        st = io.StringIO()
        with open(self.yamlfile, 'rt') as f:
            st.write(f.read())
        qc = QcConfig(st)
        st.close()
        assert qc.config == self.expected_dict


class ConfigRunTest(unittest.TestCase):

    def setUp(self):
        self.config = {
            'qartod': {
                'gross_range_test': {
                    'suspect_span': [1, 11],
                    'fail_span': [0, 12],
                }
            }
        }

    def test_run(self):
        qc = QcConfig(self.config)
        r = qc.run(
            inp=list(range(13))
        )

        expected = np.array([3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3])
        npt.assert_array_equal(
            r['qartod']['gross_range_test'],
            expected
        )
        assert 'aggregate' not in r['qartod']

    def test_run_with_agg(self):
        qc = QcConfig({'qartod': {
            'gross_range_test': {
                'fail_span': [0, 12],
            },
            'spike_test': {
                'suspect_threshold': 3,
                'fail_threshold': 10,
            }
        }})
        inp = [-1, 0, 1, 2, 10, 3]
        expected_gross_range = np.array([4, 1, 1, 1, 1, 1])
        expected_spike = np.array([2, 1, 1, 3, 3, 2])

        r = qc.run(
            inp=inp
        )

        npt.assert_array_equal(r['qartod']['gross_range_test'], expected_gross_range)
        npt.assert_array_equal(r['qartod']['spike_test'], expected_spike)

    def test_different_kwargs_run(self):

        config = deepcopy(self.config)
        config['qartod']['location_test'] = {
            'bbox': [-100, -40, 100, 40]
        }

        xs = [ -101, -100, -99, 0, 99, 100, 101 ]
        ys = [  -41,  -40, -39, 0, 39,  40,  41 ]
        qc = QcConfig(config)
        r = qc.run(
            inp=list(range(7)),
            lat=ys,
            lon=xs
        )

        range_expected = np.array([3, 1, 1, 1, 1, 1, 1])
        npt.assert_array_equal(
            r['qartod']['gross_range_test'],
            range_expected
        )
        location_expected = np.array([4, 1, 1, 1, 1, 1, 4])
        npt.assert_array_equal(
            r['qartod']['location_test'],
            location_expected
        )

    def test_with_values_in_config(self):

        config = deepcopy(self.config)
        config['qartod']['location_test'] = {
            'bbox': [-100, -40, 100, 40],
            'lat': [  -41,  -40, -39, 0, 39,  40,  41 ],
            'lon': [ -101, -100, -99, 0, 99, 100, 101 ],
        }
        config['qartod']['gross_range_test']['inp'] = list(range(7))

        qc = QcConfig(config)
        r = qc.run()

        range_expected = np.array([3, 1, 1, 1, 1, 1, 1])
        npt.assert_array_equal(
            r['qartod']['gross_range_test'],
            range_expected
        )
        location_expected = np.array([4, 1, 1, 1, 1, 1, 4])
        npt.assert_array_equal(
            r['qartod']['location_test'],
            location_expected
        )

    def test_with_empty_config(self):
        self.config['qartod']['flat_line_test'] = None
        qc = QcConfig(self.config)
        r = qc.run(
            inp=list(range(13))
        )

        assert 'gross_range_test' in r['qartod']
        assert 'flat_line_test' not in r['qartod']


class ClimatologyConfigConversionTest(unittest.TestCase):
    # Verify that we can parse and convert configs into a ClimatologyConfig object

    def setUp(self):

        # Explicitly defined config
        self.cc = ClimatologyConfig()
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
            vspan=(50, 60),
            zspan=(0, 10)
        )
        self.cc.add(
            tspan=(0, 2),
            vspan=(10, 20),
            period='month'
        )

        # JSON config, same definition as above
        self.json_config = {
            'qartod': {
                'climatology_test': {
                    'config': [
                        {
                            'vspan': (10, 20),
                            'tspan': (np.datetime64('2011-01'), np.datetime64('2011-07')),
                        },
                        {
                            'vspan': (30, 40),
                            'tspan': (np.datetime64('2011-07'), np.datetime64('2012-01')),
                        },
                        {
                            'vspan': (50, 60),
                            'zspan': (0, 10),
                            'tspan': (np.datetime64('2012-01'), np.datetime64('2013-01'))
                        },
                        {
                            'vspan': (10, 20),
                            'tspan': (0, 2),
                            'period': 'month'
                        }
                    ]
                }
            }
        }

        # YAML config, same definition as above
        template = """
        qartod:
            climatology_test:
                config:
                    - vspan: [10, 20]
                      tspan:
                        - !!timestamp 2011-01-01 00:00:00
                        - !!timestamp 2011-07-01 00:00:00
                    - vspan: [30, 40]
                      tspan:
                        - !!timestamp 2011-07-01
                        - !!timestamp 2012-01-01
                    - vspan: [50, 60]
                      zspan: [0, 10]
                      tspan:
                        - !!timestamp 2012-01-01
                        - !!timestamp 2013-01-01
                    - vspan: [10, 20]
                      tspan: [0, 2]
                      period: month
        """
        self.handle, self.yamlfile = tempfile.mkstemp(suffix='.yaml')
        with open(self.yamlfile, 'w') as f:
            f.write(template)

    def tearDown(self):
        os.close(self.handle)
        os.remove(self.yamlfile)

    def test_climatology_config_yaml_conversion(self):
        qc = QcConfig(self.yamlfile)
        yaml_climatology_config = ClimatologyConfig.convert(qc.config['qartod']['climatology_test']['config'])
        self._assert_cc_configs_equal(self.cc, yaml_climatology_config)

    def test_climatology_json_conversion(self):
        qc = QcConfig(self.json_config)
        json_climatology_config = ClimatologyConfig.convert(qc.config['qartod']['climatology_test']['config'])
        self._assert_cc_configs_equal(self.cc, json_climatology_config)

    def _assert_cc_configs_equal(self, c1: ClimatologyConfig, c2: ClimatologyConfig):
        assert len(c1.members) == len(c2.members)
        for idx in range(0, len(c1.members)):
            m1 = c1.members[idx]
            m2 = c2.members[idx]
            assert m1.tspan == m2.tspan, f"{idx} tspan did not match"
            assert m1.vspan == m2.vspan, f"{idx} vspan did not match"
            assert m1.zspan == m2.zspan, f"{idx} zspan did not match"
            assert m1.period == m2.period, f"{idx} period did not match"
