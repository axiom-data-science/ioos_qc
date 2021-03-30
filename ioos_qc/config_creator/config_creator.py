import datetime
import json
import logging
from copy import copy
from pathlib import Path

from jsonschema import validate
import numpy as np
import xarray as xr

from ioos_qc.config_creator import fx_parser

L = logging.getLogger(__name__)  # noqa


QC_CONFIG_CREATOR_SCHEMA = {
    "type": "object",
    "title": "QcConfigCreator Schema",
    "description": "Schema to validate configuration for QcCreatorConfig",
    "required": [
        "datasets"
    ],
    "definitions": {
        "variable": {
            "type": "object",
            "title": "Variable map (variable name in QcConfig -> variable name in dataset",
            "items": {"type": "object"}
        },
        "dataset": {
            "type": "object",
            "title": "Dataset description",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Name of dataset"
                },
                "file_path": {
                    "type": "string",
                    "description": "Path to file used to create configuration."
                },
                "variables": {
                    "type": "object",
                    "description": "Variables in file used to create configuration."
                },
                "3d": {
                    "type": "string",
                    "description": "Include if 3d dataset with value being the name of the 3rd dimension"
                }
            },
            "required": [
                "name",
                "file_path",
                "variables"
            ],
        },
    },
    "properties": {
        "datasets": {
            "type": "array",
            "description": "Array of datasets used for QcCreatorConfig",
            "items": {
                "$ref": "#/definitions/dataset"
            }
        }
    }
}

VARIABLE_CONFIG_SCHEMA = {
    "type": "object",
    "title": "QcVariableConfig",
    "description": "Schema to validate configuration for QcVariableConfig",
    "required": [
        "variable",
        "bbox",
        "start_time",
        "end_time",
        "tests"
    ],
    "definitions": {
        "test": {
            "type": "object",
            "description": "Test (gross_range_test, etc.) definition",
            "required": [
                "suspect_min",
                "suspect_max",
                "fail_min",
                "fail_max"
            ],
            "properties": {
                "suspect_min": {
                    "type": "string",
                    "description": "Function or value to define suspect min"
                },
                "suspect_max": {
                    "type": "string",
                    "description": "Function or value to define suspect max"
                },
                "fail_min": {
                    "type": "string",
                    "description": "Function or value to define fail min"
                },
                "fail_max": {
                    "type": "string",
                    "description": "Function or value to define fail max"
                }
            }
        }
    },
    "properties": {
        "variable": {
            "type": "string",
            "description": "Variable name. The same name used as a key in 'variable' object in CreatorConfig"
        },
        "bbox": {
            "type": "array",
            "description": "Bounding box of region in deployment (xmin, ymin, xmax, ymax)"
        },
        "start_time": {
            "type": "string",
            "description": "Start time of deployment (YYYY-MM-DD)"
        },
        "end_time": {
            "type": "string",
            "description": "Exclusive end time of deployment (YYYY-MM-DD)"
        },
        "tests": {
            "type": "object",
            "items": {
                "$ref": "#/definitions/test",
            }
        }
    }
}


class CreatorConfig(dict):
    """
    Defines the dataset(s) configuration used by QcConfigCreator.

    :param path_or_dict: QcConfigCreator configuration, one of the following formats:
        python dict
        JSON filepath (str or Path object)
    :param dict: JSON schema for CreatorConfig
    """
    def __init__(self, path_or_dict, schema=QC_CONFIG_CREATOR_SCHEMA):
        if isinstance(path_or_dict, str) or isinstance(path_or_dict, Path):
            with open(path_or_dict) as f:
                config = json.load(f)
        elif isinstance(path_or_dict, dict):
            config = path_or_dict
        else:
            raise ValueError('Input is not valid file path or dict')
        validate(instance=config, schema=schema)

        datasets = {}
        for dataset in config['datasets']:
            if '3d' in dataset:
                datasets[dataset['name']] = {
                    'file_path': dataset['file_path'],
                    'variables': dataset['variables'],
                    '3d': dataset['3d']
                }
            else:
                datasets[dataset['name']] = {
                    'file_path': dataset['file_path'],
                    'variables': dataset['variables'],
                }
        self.update(datasets)

    def __str__(self):
        return json.dumps(self, indent=4, sort_keys=True)


class QcVariableConfig(dict):
    """
    Used to generate a QcConfig for a specific variable.

    Args:
        path_or_dict: QcVariableConfig configuration, one of the following formats:
            python dict
            JSON filepath (str or Path object)
        schema: JSON schema for QcVariable
    """
    allowed_stats = [
        'min',
        'max',
        'mean',
        'std'
    ]
    allowed_operators = [
        '+',
        '-',
        '*',
        '/',
    ]
    allowed_groupings = [
        '(',
        ')'
    ]

    def __init__(self, path_or_dict, schema=VARIABLE_CONFIG_SCHEMA):
        if isinstance(path_or_dict, str) or isinstance(path_or_dict, Path):
            with open(path_or_dict) as f:
                config = json.load(f)
        elif isinstance(path_or_dict, dict):
            config = path_or_dict
        else:
            raise ValueError('Input is not valid file path or dict')

        L.debug(f"Validating schema...")
        validate(instance=config, schema=schema)

        # validate test specifications only contain allowed stats and operators
        for test in config['tests'].keys():
            for test_name, test_def in config['tests'][test].items():
                if test_name == 'bbox':
                    continue
                self._validate_fx(test_def, test_name)

        self.update(config)

    def _validate_fx(self, input_fx, test_name):
        """Thows exception if input_fx contains tokens not specifically allowed"""
        tokens = input_fx.split(' ')
        for token in tokens:
            try:
                _ = float(token)
            except ValueError:
                if token not in self.allowed_stats and \
                   token not in self.allowed_operators and \
                   token not in self.allowed_groupings:
                    msg = (
                        f'{token} not allowed in min/max specification in config of {test_name}.\n'
                        f'Allowable stats are: {[s for s in self.allowed_stats]}.\n'
                        f'Allowable operators are: {[o for o in self.allowed_operators]}.'
                        f'Allowable groupings are: {[o for o in self.allowed_groupings]}.'
                    )
                    raise ValueError(msg)

    def __str__(self):
        return json.dumps(self, indent=4, sort_keys=True)


class QcConfigCreator:
    """Creates level-0 configuration to create QcQonfig.

    Arguments:
        creator_config (QcCreatorConfig): Configuration for datasets and variables used to create qc_config.

    Attributes:
        allowed_stats (list): Specific statistics allowed to be used to configure each test.
        allowed_operators (list): Operators allowed to used to configure each test.
    """

    def __init__(self, creator_config):
        self.config = creator_config
        self.datasets = self._load_datasets()
        self.dataset_years = self._determine_dataset_years()

    def create_config(self, variable_config):
        """Create QARTOD QC config given QcVariableConfig.

        Args:
            variable_config (QcVariableConfig): Config for variable to be quality controlled

        Returns:
            qc_config (dict): Config for ioos_qc
        """
        stats = self._get_stats(variable_config)
        L.debug("Creating config...")
        test_configs = {
            name: self._create_test_section(name, variable_config, stats) for name in variable_config['tests'].keys()
        }

        return {
            f'{variable_config["variable"]}': {
                'qartod': test_configs
            }
        }

    def _load_datasets(self):
        """Load datasets"""
        L.debug(f"Loading {len(self.config)} datasets...")
        return {name: xr.load_dataset(self.config[name]['file_path']) for name in self.config.keys()}

    def _determine_dataset_years(self):
        """Determine year used in datasets, return as dict {dataset_name, year}.

        Notes:
            - Each dataset is from a unique climatology or source,
              so the monthly files have different years.
        """
        years = {}
        for dataset_name, dataset in self.datasets.items():
            years[dataset_name] = dataset['time.year'][0].data

        return years

    def _var2var_in_file(self, var):
        """Return variable name used in the dataset and dataset name"""
        for dataset_name, dataset in self.config.items():
            if var in dataset['variables'].keys():
                return dataset['variables'][var], dataset_name

    def var2dataset(self, var):
        """Return dataset name and dataset for given variable (as named in qc_config, not in the file)"""
        _, dataset_name = self._var2var_in_file(var)

        return dataset_name, self.datasets[dataset_name]

    def _create_test_section(self, test_name, variable_config, test_limits):
        """Given test_name, QcVariableConfig and test_limits, return qc_config section for that test."""
        if test_name == 'spike_test':
            return self.__create_spike_section(test_name, variable_config, test_limits)
        elif test_name == 'location_test':
            return self.__create_location_section(test_name, variable_config)
        elif test_name == 'rate_of_change_test':
            return self.__create_rate_of_change_section(test_name, variable_config, test_limits)
        elif test_name == 'flat_line_test':
            return self.__create_flat_line_section(test_name, variable_config, test_limits)
        else:
            return self.__create_span_section(test_name, variable_config, test_limits)

    def __create_span_section(self, test_name, variable_config, stats):
        suspect_min = fx_parser.eval_fx(variable_config['tests'][test_name]['suspect_min'], stats)
        suspect_max = fx_parser.eval_fx(variable_config['tests'][test_name]['suspect_max'], stats)
        fail_min = fx_parser.eval_fx(variable_config['tests'][test_name]['fail_min'], stats)
        fail_max = fx_parser.eval_fx(variable_config['tests'][test_name]['fail_max'], stats)

        return {
            'suspect_span': [suspect_min, suspect_max],
            'fail_span': [fail_min, fail_max]
        }

    def __create_spike_section(self, test_name, variable_config, stats):
        suspect_threshold = fx_parser.eval_fx(variable_config['tests'][test_name]['suspect_threshold'], stats)
        fail_threshold = fx_parser.eval_fx(variable_config['tests'][test_name]['fail_threshold'], stats)

        return {
            'suspect_threshold': suspect_threshold,
            'fail_threshold': fail_threshold
        }

    def __create_flat_line_section(self, test_name, variable_config, stats):
        suspect_threshold = fx_parser.eval_fx(variable_config['tests'][test_name]['suspect_threshold'], stats)
        fail_threshold = fx_parser.eval_fx(variable_config['tests'][test_name]['fail_threshold'], stats)
        tolerance = fx_parser.eval_fx(variable_config['tests'][test_name]['tolerance'], stats)

        return {
            'suspect_threshold': suspect_threshold,
            'fail_threshold': fail_threshold,
            'tolerance': tolerance
        }

    def __create_location_section(self, test_name, variable_config):
        return {
            'bbox': variable_config['tests'][test_name]['bbox']
        }

    def __create_rate_of_change_section(self, test_name, variable_config, stats):
        threshold = fx_parser.eval_fx(variable_config['tests'][test_name]['threshold'], stats)
        return {
            'threshold': threshold
        }

    def _get_stats(self, variable_config):
        """Return dict of stats (min, max, mean, std) for given config"""
        start_time = datetime.datetime.strptime(variable_config['start_time'], '%Y-%m-%d')
        end_time = datetime.datetime.strptime(variable_config['end_time'], '%Y-%m-%d')
        time_range = self._get_time_range(
            variable_config['variable'],
            (start_time, end_time)
        )
        subset = self._get_subset(
            variable_config['variable'],
            variable_config['bbox'],
            time_range
        )

        return {
            'min': np.nanmin(subset),
            'max': np.nanmax(subset),
            'mean': np.nanmean(subset),
            'std': np.nanstd(subset)
        }

    def _get_time_range(self, var, time_range):
        """Return time slice to query data from input time_range.

        Notes:
        - The source datasets have different times because they are from different climatologies.
        - This method extracts the dates, but corrects the year for the appropriate dataset.
        """
        _, name = self._var2var_in_file(var)
        year = self.dataset_years[name]

        # TODO: Support time_range that spans multiple years.
        # For example: https://ferret.pmel.noaa.gov/pmel/erddap
        # Dataset: sd1054

        start_time = f"{year}-{time_range[0].month}-{time_range[0].day}"
        end_time = f"{year}-{time_range[1].month}-{time_range[1].day}"

        L.debug(f"Calculating time range betwen {start_time} and {end_time}..")
        return slice(start_time, end_time)

    def _get_subset(self, var, bbox, time_slice, depth=0, pad_delta=0.5):
        """Get subset of data"""
        ds_name, ds = self.var2dataset(var)

        lat_mask = np.logical_and(
            ds['lat'] >= bbox[1],
            ds['lat'] <= bbox[3]
        )
        lon_mask = np.logical_and(
            ds['lon'] >= bbox[0],
            ds['lon'] <= bbox[2]
        )

        # if there is no data in the subset, increase bounding box in an iterative fashion
        # - both are interpolated to daily values
        L.debug(f"Subsetting {var} by depth={depth} and {bbox}...")
        subset = self.__get_daily_interp_subset(var, time_slice, depth, lat_mask, lon_mask)

        padded = 0
        while np.nansum(subset) == 0:
            old_bbox = copy(bbox)
            bbox = self.__apply_bbox_pad(bbox, pad_delta)

            if old_bbox == bbox:
                L.warning(f"No data found for {var} at depth={depth}")
                break

            lat_mask = np.logical_and(
                ds['lat'] >= bbox[1],
                ds['lat'] <= bbox[3]
            )
            lon_mask = np.logical_and(
                ds['lon'] >= bbox[0],
                ds['lon'] <= bbox[2]
            )

            padded += 1
            subset = self.__get_daily_interp_subset(var, time_slice, depth, lat_mask, lon_mask)

        L.info(f'Used {bbox} for bounds after padding {padded} times')
        return subset

    def __apply_bbox_pad(self, bbox, pad):
        def apply_pad(val, lat_or_lon, min_or_max):
            if lat_or_lon == 'lat':
                if min_or_max == 'min':
                    return max(-90, val - pad)
                else:
                    return min(90, val + pad)
            else:
                if val <= 0 and min_or_max == 'min':
                    return max(-180, val - pad)
                elif val <= 0 and min_or_max == 'max':
                    return min(180, val + pad)
                elif val > 0 and min_or_max == 'min':
                    return max(-180, val - pad)
                else:
                    return min(180, val + pad)

        new_bbox = []
        new_bbox.append(apply_pad(bbox[0], 'lon', 'min'))
        new_bbox.append(apply_pad(bbox[1], 'lat', 'min'))
        new_bbox.append(apply_pad(bbox[2], 'lon', 'max'))
        new_bbox.append(apply_pad(bbox[3], 'lat', 'max'))

        return new_bbox

    def __get_daily_interp_subset(self, var, time_slice, depth, lat_mask, lon_mask):
        ds_name, ds = self.var2dataset(var)
        var_in_file, _ = self._var2var_in_file(var)

        if '3d' in self.config[ds_name]:
            return ds[var_in_file][:, depth, lat_mask, lon_mask].resample(time='1D').interpolate().sel(time=time_slice)
        else:
            return ds[var_in_file][:, lat_mask, lon_mask].resample(time='1D').interpolate().sel(time=time_slice)

    def __str__(self):
        return json.dumps(self.config, indent=4, sort_keys=True)

    def __rpr__(self):
        return self.__str__


def to_json(qc_config, out_file=None):
    """Given qc_config return json"""
    if out_file:
        with open(out_file, 'w') as outfile:
            json.dump(outfile, qc_config)
    else:
        return json.dumps(qc_config)
