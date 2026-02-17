"""Shared pytest fixtures and dependency mocking for tinyml-modelmaker tests.

The tinyml_modelmaker package eagerly imports tinyml_tinyverse and tinyml_torchmodelopt
through its module init chain. These sibling packages may not be installed in the test
environment (they require Python 3.10 and heavy ML dependencies). We mock them here
before any tinyml_modelmaker imports occur.
"""

import sys
import types
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Mock external packages that tinyml_modelmaker eagerly imports but that may
# not be available in the test environment.
# ---------------------------------------------------------------------------

_PACKAGES_TO_MOCK = [
    # tinyml_tinyverse and all sub-paths imported by modelmaker training modules
    'tinyml_tinyverse',
    'tinyml_tinyverse.references',
    'tinyml_tinyverse.references.timeseries_classification',
    'tinyml_tinyverse.references.timeseries_classification.test_onnx',
    'tinyml_tinyverse.references.timeseries_classification.train',
    'tinyml_tinyverse.references.timeseries_regression',
    'tinyml_tinyverse.references.timeseries_regression.test_onnx',
    'tinyml_tinyverse.references.timeseries_regression.train',
    'tinyml_tinyverse.references.timeseries_anomalydetection',
    'tinyml_tinyverse.references.timeseries_anomalydetection.test_onnx',
    'tinyml_tinyverse.references.timeseries_anomalydetection.test_onnx_cls',
    'tinyml_tinyverse.references.timeseries_anomalydetection.train',
    'tinyml_tinyverse.references.timeseries_forecasting',
    'tinyml_tinyverse.references.timeseries_forecasting.test_onnx',
    'tinyml_tinyverse.references.timeseries_forecasting.train',
    'tinyml_tinyverse.references.image_classification',
    'tinyml_tinyverse.references.image_classification.test_onnx',
    'tinyml_tinyverse.references.image_classification.train',
    'tinyml_tinyverse.references.common',
    'tinyml_tinyverse.references.common.compilation',
    # tinyml_torchmodelopt quantization package
    'tinyml_torchmodelopt',
    'tinyml_torchmodelopt.quantization',
]

for pkg_name in _PACKAGES_TO_MOCK:
    if pkg_name not in sys.modules:
        sys.modules[pkg_name] = MagicMock()

# Ensure the quantization mock has the specific constants that params.py imports
_quant_mock = sys.modules['tinyml_torchmodelopt.quantization']
_quant_mock.TinyMLQuantizationVersion = type('TinyMLQuantizationVersion', (), {
    'NO_QUANTIZATION': 0,
    'QUANTIZATION_GENERIC': 1,
    'QUANTIZATION_TINPU': 2,
})
_quant_mock.TinyMLQuantizationMethod = type('TinyMLQuantizationMethod', (), {
    'QAT': 'qat',
    'PTQ': 'ptq',
})

# ---------------------------------------------------------------------------
# Now it is safe to import from tinyml_modelmaker
# ---------------------------------------------------------------------------

import json
import os

import pytest
import yaml


@pytest.fixture
def sample_config_dict():
    """Return a nested dict suitable for ConfigDict tests."""
    return {
        'a': 1,
        'b': {
            'c': 2,
            'd': 'hello',
        },
        'e': [1, 2, 3],
        'f': True,
    }


@pytest.fixture
def tmp_yaml_file(tmp_path, sample_config_dict):
    """Create a temporary YAML file with sample config and return its path."""
    yaml_path = tmp_path / 'config.yaml'
    with open(yaml_path, 'w') as fp:
        yaml.safe_dump(sample_config_dict, fp)
    return str(yaml_path)


@pytest.fixture
def tmp_data_dir(tmp_path):
    """Create a temporary dataset directory structure with class folders and CSV files.

    Structure:
        data/
            classA/
                file1.csv  (header + 20 rows)
                file2.csv
                file3.csv
            classB/
                file4.csv
                file5.csv
                file6.csv
    """
    data_dir = tmp_path / 'data'
    for class_name in ['classA', 'classB']:
        class_dir = data_dir / class_name
        class_dir.mkdir(parents=True)
        for i in range(3):
            file_path = class_dir / f'file_{class_name}_{i}.csv'
            # Write some CSV-like data (header + rows)
            lines = ['col1,col2,col3\n']
            for row in range(20):
                lines.append(f'{row},{row * 2},{row * 3}\n')
            file_path.write_text(''.join(lines))
    return str(data_dir)


@pytest.fixture
def sample_coco_dataset():
    """Return a minimal COCO-format dataset dict."""
    return {
        'info': {'description': 'test dataset'},
        'categories': [
            {'id': 1, 'name': 'cat', 'supercategory': 'animal'},
            {'id': 2, 'name': 'dog', 'supercategory': 'animal'},
        ],
        'images': [
            {'id': 1, 'file_name': 'img001.jpg', 'width': 100, 'height': 100},
            {'id': 2, 'file_name': 'img002.jpg', 'width': 100, 'height': 100},
            {'id': 3, 'file_name': 'img003.jpg', 'width': 100, 'height': 100},
            {'id': 4, 'file_name': 'img004.jpg', 'width': 100, 'height': 100},
            {'id': 5, 'file_name': 'img005.jpg', 'width': 100, 'height': 100},
            {'id': 6, 'file_name': 'img006.jpg', 'width': 100, 'height': 100},
            {'id': 7, 'file_name': 'img007.jpg', 'width': 100, 'height': 100},
            {'id': 8, 'file_name': 'img008.jpg', 'width': 100, 'height': 100},
            {'id': 9, 'file_name': 'img009.jpg', 'width': 100, 'height': 100},
            {'id': 10, 'file_name': 'img010.jpg', 'width': 100, 'height': 100},
        ],
        'annotations': [
            {'id': 1, 'image_id': 1, 'category_id': 1, 'bbox': [10, 10, 50, 50]},
            {'id': 2, 'image_id': 2, 'category_id': 1, 'bbox': [20, 20, 40, 40]},
            {'id': 3, 'image_id': 3, 'category_id': 2, 'bbox': [15, 15, 45, 45]},
            {'id': 4, 'image_id': 4, 'category_id': 2, 'bbox': [10, 10, 50, 50]},
            {'id': 5, 'image_id': 5, 'category_id': 1, 'bbox': [20, 20, 40, 40]},
            {'id': 6, 'image_id': 6, 'category_id': 1, 'bbox': [15, 15, 45, 45]},
            {'id': 7, 'image_id': 7, 'category_id': 2, 'bbox': [10, 10, 50, 50]},
            {'id': 8, 'image_id': 8, 'category_id': 2, 'bbox': [20, 20, 40, 40]},
            {'id': 9, 'image_id': 9, 'category_id': 1, 'bbox': [15, 15, 45, 45]},
            {'id': 10, 'image_id': 10, 'category_id': 1, 'bbox': [10, 10, 50, 50]},
        ],
    }
