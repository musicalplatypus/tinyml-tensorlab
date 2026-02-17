"""Tests for tinyml_modelmaker.ai_modules.timeseries.params.init_params."""

import pytest

from tinyml_modelmaker.utils.config_dict import ConfigDict
from tinyml_modelmaker.ai_modules.timeseries.params import init_params


class TestInitParams:
    """Tests for the init_params function."""

    def test_returns_configdict(self):
        params = init_params()
        assert isinstance(params, ConfigDict)

    def test_has_required_sections(self):
        params = init_params()
        required = ['common', 'dataset', 'training', 'testing',
                     'data_processing_feature_extraction', 'compilation']
        for section in required:
            assert section in params, f'Missing section: {section}'

    def test_default_split_names(self):
        params = init_params()
        assert params.dataset.split_names == ('train', 'val', 'test')

    def test_default_split_factor(self):
        params = init_params()
        assert params.dataset.split_factor == (0.6, 0.3, 0.1)

    def test_override_common(self):
        params = init_params({'common': {'verbose_mode': False}})
        assert params.common.verbose_mode is False

    def test_nested_override(self):
        params = init_params({'training': {'batch_size': 32}})
        assert params.training.batch_size == 32

    def test_training_defaults(self):
        params = init_params()
        assert isinstance(params.training.training_epochs, int)
        assert params.training.training_epochs > 0
        assert isinstance(params.training.batch_size, int)
        assert params.training.batch_size > 0
        assert isinstance(params.training.learning_rate, float)
        assert params.training.learning_rate > 0

    def test_dataset_defaults(self):
        params = init_params()
        assert params.dataset.enable is True
        assert params.dataset.data_dir == 'classes'
        assert params.dataset.annotation_dir == 'annotations'
        assert params.dataset.split_type == 'amongst_files'

    def test_compilation_defaults(self):
        params = init_params()
        assert params.compilation.enable is True
