"""Tests for the BaseModelTraining base class and module-level helper functions.

Verifies that the base class provides correct defaults, that hook methods
work as expected, and that the module-level get_model_descriptions/
get_model_description helpers filter correctly.
"""

import threading
from unittest.mock import MagicMock

from tinyml_modelmaker.ai_modules.timeseries.training.tinyml_tinyverse.base_training import (
    BaseModelTraining,
    get_model_descriptions,
    get_model_description,
)


# ===================================================================
# get_model_descriptions / get_model_description helpers
# ===================================================================


class TestGetModelDescriptions:
    """Tests for the module-level model description helpers."""

    def test_filters_by_enabled_list(self):
        descriptions = {'a': {'v': 1}, 'b': {'v': 2}, 'c': {'v': 3}}
        enabled = ['a', 'c']
        result = get_model_descriptions(descriptions, enabled)
        assert result == {'a': {'v': 1}, 'c': {'v': 3}}

    def test_returns_empty_when_none_enabled(self):
        descriptions = {'a': {'v': 1}, 'b': {'v': 2}}
        enabled = []
        result = get_model_descriptions(descriptions, enabled)
        assert result == {}

    def test_ignores_enabled_names_not_in_descriptions(self):
        descriptions = {'a': {'v': 1}}
        enabled = ['a', 'missing']
        result = get_model_descriptions(descriptions, enabled)
        assert result == {'a': {'v': 1}}

    def test_get_model_description_found(self):
        descriptions = {'model_a': {'detail': 'info'}}
        enabled = ['model_a']
        result = get_model_description(descriptions, enabled, 'model_a')
        assert result == {'detail': 'info'}

    def test_get_model_description_missing_returns_none(self):
        descriptions = {'model_a': {'detail': 'info'}}
        enabled = ['model_a']
        result = get_model_description(descriptions, enabled, 'no_such_model')
        assert result is None

    def test_get_model_description_disabled_returns_none(self):
        descriptions = {'model_a': {'detail': 'info'}}
        enabled = []  # model_a is in descriptions but not enabled
        result = get_model_description(descriptions, enabled, 'model_a')
        assert result is None


# ===================================================================
# BaseModelTraining class
# ===================================================================


class TestBaseModelTraining:
    """Tests for the BaseModelTraining abstract base class."""

    def test_build_train_argv_raises_not_implemented(self):
        """Direct use of _build_train_argv should raise NotImplementedError."""
        try:
            BaseModelTraining._build_train_argv(None, 'cpu', 0)
            assert False, "Expected NotImplementedError"
        except NotImplementedError:
            pass

    def test_build_test_argv_raises_not_implemented(self):
        """Direct use of _build_test_argv should raise NotImplementedError."""
        try:
            BaseModelTraining._build_test_argv(None, 'cpu', '/data', '/model', '/out')
            assert False, "Expected NotImplementedError"
        except NotImplementedError:
            pass

    def test_default_quant_min_epochs(self):
        """Default _get_quant_min_epochs should return 10."""
        obj = object.__new__(BaseModelTraining)
        assert obj._get_quant_min_epochs() == 10

    def test_default_quant_extra_argv(self):
        """Default _get_quant_extra_argv should return empty list."""
        obj = object.__new__(BaseModelTraining)
        assert obj._get_quant_extra_argv() == []

    def test_default_extra_init_params(self):
        """Default _get_extra_init_params should return empty dict."""
        obj = object.__new__(BaseModelTraining)
        assert obj._get_extra_init_params() == {}

    def test_default_post_process_test_args_passthrough(self):
        """Default _post_process_test_args should return args unchanged."""
        obj = object.__new__(BaseModelTraining)
        sentinel = object()
        assert obj._post_process_test_args(sentinel) is sentinel

    def test_stop_with_quit_event(self):
        """stop() should set the quit_event and return True."""
        obj = object.__new__(BaseModelTraining)
        obj.quit_event = threading.Event()
        assert obj.stop() is True
        assert obj.quit_event.is_set()

    def test_stop_without_quit_event(self):
        """stop() should return False when quit_event is None."""
        obj = object.__new__(BaseModelTraining)
        obj.quit_event = None
        assert obj.stop() is False

    def test_get_params_returns_params(self):
        """get_params() should return the params attribute."""
        obj = object.__new__(BaseModelTraining)
        obj.params = {'test': 'value'}
        assert obj.get_params() == {'test': 'value'}

    def test_init_params_returns_config_dict(self):
        """init_params should return a ConfigDict with training key."""
        from tinyml_modelmaker import utils
        result = BaseModelTraining.init_params()
        assert isinstance(result, utils.ConfigDict)


# ===================================================================
# Subclass override verification
# ===================================================================


class TestSubclassOverrides:
    """Verify that subclass override hooks work correctly."""

    def test_quant_min_epochs_override(self):
        """Subclass can override _get_quant_min_epochs."""

        class RegressionTraining(BaseModelTraining):
            _train_module = MagicMock()
            _test_module = MagicMock()

            def _get_quant_min_epochs(self):
                return 50

            def _build_train_argv(self, device, distributed):
                return []

            def _build_test_argv(self, device, data_path, model_path, output_dir):
                return []

        obj = object.__new__(RegressionTraining)
        assert obj._get_quant_min_epochs() == 50

    def test_quant_extra_argv_override(self):
        """Subclass can override _get_quant_extra_argv."""

        class RegressionTraining(BaseModelTraining):
            _train_module = MagicMock()
            _test_module = MagicMock()

            def _get_quant_extra_argv(self):
                return ['--lambda-reg', '0.1']

            def _build_train_argv(self, device, distributed):
                return []

            def _build_test_argv(self, device, data_path, model_path, output_dir):
                return []

        obj = object.__new__(RegressionTraining)
        assert obj._get_quant_extra_argv() == ['--lambda-reg', '0.1']

    def test_post_process_test_args_override(self):
        """Subclass can override _post_process_test_args to set extra attrs."""

        class ADTraining(BaseModelTraining):
            _train_module = MagicMock()
            _test_module = MagicMock()

            def _post_process_test_args(self, args):
                args.cache_dataset = None
                args.gpu = 2
                return args

            def _build_train_argv(self, device, distributed):
                return []

            def _build_test_argv(self, device, data_path, model_path, output_dir):
                return []

        obj = object.__new__(ADTraining)
        args = MagicMock()
        result = obj._post_process_test_args(args)
        assert result.cache_dataset is None
        assert result.gpu == 2

    def test_extra_init_params_override(self):
        """Subclass can override _get_extra_init_params."""

        class ClassificationTraining(BaseModelTraining):
            _train_module = MagicMock()
            _test_module = MagicMock()

            def _get_extra_init_params(self):
                return dict(file_level_classification_log_path='/tmp/test.log')

            def _build_train_argv(self, device, distributed):
                return []

            def _build_test_argv(self, device, data_path, model_path, output_dir):
                return []

        obj = object.__new__(ClassificationTraining)
        extra = obj._get_extra_init_params()
        assert extra == {'file_level_classification_log_path': '/tmp/test.log'}
