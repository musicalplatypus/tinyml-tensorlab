"""Tests for tinyml_modelmaker constants modules."""

import pytest

from tinyml_modelmaker.ai_modules.timeseries import constants as ts_constants
from tinyml_modelmaker.ai_modules.vision import constants as vis_constants


class TestTimeseriesConstants:
    """Tests for timeseries constants."""

    def test_task_types_are_strings(self):
        task_types = [
            ts_constants.TASK_TYPE_MOTOR_FAULT,
            ts_constants.TASK_TYPE_ARC_FAULT,
            ts_constants.TASK_TYPE_GENERIC_TS_CLASSIFICATION,
            ts_constants.TASK_TYPE_GENERIC_TS_REGRESSION,
            ts_constants.TASK_TYPE_GENERIC_TS_ANOMALYDETECTION,
            ts_constants.TASK_TYPE_GENERIC_TS_FORECASTING,
        ]
        for tt in task_types:
            assert isinstance(tt, str), f'{tt} should be a string'

    def test_task_categories_are_strings(self):
        categories = [
            ts_constants.TASK_CATEGORY_TS_CLASSIFICATION,
            ts_constants.TASK_CATEGORY_TS_REGRESSION,
            ts_constants.TASK_CATEGORY_TS_ANOMALYDETECTION,
            ts_constants.TASK_CATEGORY_TS_FORECASTING,
        ]
        for cat in categories:
            assert isinstance(cat, str), f'{cat} should be a string'

    def test_target_devices_no_duplicates(self):
        devices = ts_constants.TARGET_DEVICES_ALL
        assert len(devices) == len(set(devices)), 'TARGET_DEVICES_ALL should have no duplicates'

    def test_target_devices_are_strings(self):
        for device in ts_constants.TARGET_DEVICES_ALL:
            assert isinstance(device, str), f'{device} should be a string'

    def test_split_names_default(self):
        assert ts_constants.SPLIT_NAMES_DEFAULT == ('train', 'val', 'test')

    def test_split_name_constants(self):
        assert ts_constants.SPLIT_NAME_TRAIN == 'train'
        assert ts_constants.SPLIT_NAME_VAL == 'val'
        assert ts_constants.SPLIT_NAME_TEST == 'test'

    def test_training_backend_constant(self):
        assert ts_constants.TRAINING_BACKEND_TINYML_TINYVERSE == 'tinyml_tinyverse'

    def test_target_module_constant(self):
        assert ts_constants.TARGET_MODULE_TIMESERIES == 'timeseries'

    def test_task_descriptions_have_required_keys(self):
        for task_key, desc in ts_constants.TASK_DESCRIPTIONS.items():
            assert 'task_name' in desc, f'{task_key} missing task_name'
            assert 'target_module' in desc, f'{task_key} missing target_module'
            assert 'target_devices' in desc, f'{task_key} missing target_devices'
            assert 'stages' in desc, f'{task_key} missing stages'


class TestVisionConstants:
    """Tests for vision constants."""

    def test_task_type_is_string(self):
        assert isinstance(vis_constants.TASK_TYPE_IMAGE_CLASSIFICATION, str)

    def test_task_category_is_string(self):
        assert isinstance(vis_constants.TASK_CATEGORY_IMAGE_CLASSIFICATION, str)

    def test_target_devices_no_duplicates(self):
        devices = vis_constants.TARGET_DEVICES_ALL
        assert len(devices) == len(set(devices)), 'TARGET_DEVICES_ALL should have no duplicates'

    def test_split_names_default(self):
        assert vis_constants.SPLIT_NAMES_DEFAULT == ('train', 'val', 'test')

    def test_training_backend_constant(self):
        assert vis_constants.TRAINING_BACKEND_TINYML_TINYVERSE == 'tinyml_tinyverse'

    def test_target_module_constant(self):
        assert vis_constants.TARGET_MODULE_VISION == 'vision'

    def test_task_descriptions_have_required_keys(self):
        for task_key, desc in vis_constants.TASK_DESCRIPTIONS.items():
            assert 'task_name' in desc, f'{task_key} missing task_name'
            assert 'target_module' in desc, f'{task_key} missing target_module'
            assert 'target_devices' in desc, f'{task_key} missing target_devices'
            assert 'stages' in desc, f'{task_key} missing stages'


class TestCrossModuleConsistency:
    """Tests for consistency between timeseries and vision constants."""

    def test_target_devices_match(self):
        """Both modules should define the same set of target devices."""
        ts_devices = set(ts_constants.TARGET_DEVICES_ALL)
        vis_devices = set(vis_constants.TARGET_DEVICES_ALL)
        assert ts_devices == vis_devices, (
            f'Device mismatch: only in timeseries={ts_devices - vis_devices}, '
            f'only in vision={vis_devices - ts_devices}'
        )

    def test_split_names_match(self):
        assert ts_constants.SPLIT_NAMES_DEFAULT == vis_constants.SPLIT_NAMES_DEFAULT
