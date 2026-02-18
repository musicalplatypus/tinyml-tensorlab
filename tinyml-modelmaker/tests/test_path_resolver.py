"""Tests for the resolve_paths() and resolve_run_name() utilities (improvement 3.3).

These functions were extracted from the duplicated ModelRunner.__init__()
path resolution logic in both timeseries/runner.py and vision/runner.py.
"""

import os

import pytest

from tinyml_modelmaker.utils.config_dict import ConfigDict
from tinyml_modelmaker.utils.misc_utils import resolve_paths, resolve_run_name

# The mock in conftest.py sets NO_QUANTIZATION = 0
_NO_QUANTIZATION = 0
_QUANTIZATION_GENERIC = 1

TARGET_DEVICES = ['F28P55', 'F28P65', 'MSPM0G3507']


def _make_params(**overrides):
    """Build a minimal ConfigDict with the fields resolve_paths() needs."""
    base = {
        'common': {
            'projects_path': '/tmp/projects',
            'run_name': 'test_run',
            'target_device': 'F28P55',
        },
        'dataset': {
            'dataset_name': 'my_dataset',
            'dataset_path': '/tmp/dataset',
            'input_data_path': 'data/input.zip',
            'input_annotation_path': 'data/annotations.json',
        },
        'training': {
            'train_output_path': None,
            'model_name': 'test_model',
            'quantization': _NO_QUANTIZATION,
            'enable': True,
            'target_devices': {},
        },
        'compilation': {
            'compile_output_path': None,
            'enable': False,
        },
    }
    # Apply overrides via deep merge
    params = ConfigDict(base)
    if overrides:
        params.update(overrides)
    return params


class TestResolveRunName:
    """Tests for the resolve_run_name() helper."""

    def test_empty_run_name_returns_empty_string(self):
        assert resolve_run_name('', 'some_model') == ''

    def test_none_run_name_returns_empty_string(self):
        assert resolve_run_name(None, 'some_model') == ''

    def test_model_name_substitution(self):
        result = resolve_run_name('{model_name}_run', 'my_model')
        assert result == 'my_model_run'

    def test_datetime_substitution(self):
        result = resolve_run_name('{date-time}_run', 'model')
        # Should be YYYYMMDD-HHMMSS_run
        assert result.endswith('_run')
        assert len(result) == len('20260218-120000_run')
        assert '-' in result

    def test_plain_name_unchanged(self):
        assert resolve_run_name('my_run', 'model') == 'my_run'


class TestResolvePaths:
    """Tests for the resolve_paths() utility."""

    def test_default_paths(self):
        """Default mode: nested structure under projects_path/dataset_name."""
        params = _make_params()
        resolve_paths(params, TARGET_DEVICES)

        assert os.path.isabs(params.common.projects_path)
        assert params.common.project_path.endswith(os.path.join('projects', 'my_dataset'))
        assert params.common.project_run_path.endswith(
            os.path.join('my_dataset', 'run', 'test_run'))
        assert params.dataset.dataset_path.endswith(
            os.path.join('my_dataset', 'dataset'))
        assert 'training' in params.training.training_path
        assert params.training.model_packaged_path.endswith('.zip')

    def test_custom_train_output_path(self):
        """With train_output_path: flat structure under that directory."""
        params = _make_params(training={'train_output_path': '/tmp/custom_out',
                                         'model_name': 'test_model',
                                         'quantization': _NO_QUANTIZATION,
                                         'enable': True,
                                         'target_devices': {}})
        resolve_paths(params, TARGET_DEVICES)

        assert params.common.projects_path == '/tmp/custom_out'
        assert params.common.project_run_path == '/tmp/custom_out'
        assert params.dataset.dataset_path == os.path.join('/tmp/custom_out', 'dataset')
        assert 'training_base' in params.training.training_path

    def test_custom_compile_output_path(self):
        """With compile_output_path: compilation goes to that directory."""
        params = _make_params(compilation={'compile_output_path': '/tmp/compile_out',
                                            'enable': True})
        resolve_paths(params, TARGET_DEVICES)

        assert params.compilation.compilation_path == '/tmp/compile_out'
        assert 'F28P55' in params.compilation.model_packaged_path
        assert params.compilation.model_packaged_path.endswith('.zip')

    def test_compile_only_mode(self):
        """When training disabled + compilation enabled with custom path."""
        params = _make_params(
            training={'train_output_path': None, 'model_name': 'test_model',
                      'quantization': _NO_QUANTIZATION, 'enable': False,
                      'target_devices': {}},
            compilation={'compile_output_path': '/tmp/compile_out', 'enable': True})
        resolve_paths(params, TARGET_DEVICES)

        # In compile-only mode, projects_path should be overridden to compile_output_path
        assert params.common.projects_path == '/tmp/compile_out'
        assert params.common.project_run_path == '/tmp/compile_out'

    def test_dataset_name_from_input_path(self):
        """Empty dataset_name should be derived from input_data_path."""
        params = _make_params(dataset={'dataset_name': '',
                                        'dataset_path': '/tmp/dataset',
                                        'input_data_path': '/data/my_archive.zip',
                                        'input_annotation_path': None})
        resolve_paths(params, TARGET_DEVICES)

        assert params.dataset.dataset_name == 'my_archive'

    def test_invalid_target_device_raises_valueerror(self):
        params = _make_params(common={'projects_path': '/tmp/projects',
                                       'run_name': 'run',
                                       'target_device': 'NONEXISTENT_DEVICE'})
        with pytest.raises(ValueError, match='common.target_device must be set to one of'):
            resolve_paths(params, TARGET_DEVICES)

    def test_run_name_template_expansion(self):
        """Run name with {model_name} placeholder should be expanded."""
        params = _make_params(common={'projects_path': '/tmp/projects',
                                       'run_name': '{model_name}_experiment',
                                       'target_device': 'F28P55'})
        resolve_paths(params, TARGET_DEVICES)

        assert params.common.run_name == 'test_model_experiment'

    def test_quantization_path_set_when_enabled(self):
        """Quantization path should be set when quantization is not NO_QUANTIZATION."""
        params = _make_params(training={'train_output_path': None,
                                         'model_name': 'test_model',
                                         'quantization': _QUANTIZATION_GENERIC,
                                         'enable': True,
                                         'target_devices': {}})
        resolve_paths(params, TARGET_DEVICES)

        assert hasattr(params.training, 'training_path_quantization')
        assert 'quantization' in params.training.training_path_quantization

    def test_quantization_path_not_set_when_disabled(self):
        """Quantization path should not be created when quantization is NO_QUANTIZATION."""
        params = _make_params(training={'train_output_path': None,
                                         'model_name': 'test_model',
                                         'quantization': _NO_QUANTIZATION,
                                         'enable': True,
                                         'target_devices': {}})
        resolve_paths(params, TARGET_DEVICES)

        # training_path_quantization should not have been set
        assert not hasattr(params.training, 'training_path_quantization') or \
               params.training.get('training_path_quantization') is None

    def test_paths_are_absolute(self):
        """All resolved paths should be absolute."""
        params = _make_params()
        resolve_paths(params, TARGET_DEVICES)

        assert os.path.isabs(params.common.projects_path)
        assert os.path.isabs(params.common.project_path)
        assert os.path.isabs(params.common.project_run_path)
        assert os.path.isabs(params.dataset.dataset_path)
        assert os.path.isabs(params.training.training_path)
        assert os.path.isabs(params.compilation.compilation_path)
