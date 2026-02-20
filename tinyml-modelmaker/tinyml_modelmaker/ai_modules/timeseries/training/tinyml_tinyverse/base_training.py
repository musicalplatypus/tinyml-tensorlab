#################################################################################
# Copyright (c) 2023-2024, Texas Instruments
# All Rights Reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#################################################################################

"""Base class for timeseries model training.

All four timeseries task types (classification, regression, anomaly detection,
forecasting) share the same training lifecycle:  ``__init__`` sets up logging
regex patterns and paths, ``run()`` builds CLI argv, launches training, handles
quantization retraining, and runs ONNX testing.

This base class captures that shared logic.  Subclasses override
``_build_train_argv`` and ``_build_test_argv`` to supply the task-specific
command-line arguments.
"""

import os
import shutil

import torch.backends.mps

from tinyml_torchmodelopt.quantization import TinyMLQuantizationVersion

import tinyml_modelmaker

from ..... import utils


# ---------------------------------------------------------------------------
# Module-level helper functions (shared across all task types)
# ---------------------------------------------------------------------------

def get_model_descriptions(_model_descriptions, enabled_models_list, task_type=None):
    """Return model descriptions filtered by the enabled models list."""
    return {k: v for k, v in _model_descriptions.items() if k in enabled_models_list}


def get_model_description(_model_descriptions, enabled_models_list, model_name):
    """Return a single model description by name, or None if not found."""
    descs = get_model_descriptions(_model_descriptions, enabled_models_list)
    return descs[model_name] if model_name in descs else None


# ---------------------------------------------------------------------------
# Base training class
# ---------------------------------------------------------------------------

class BaseModelTraining:
    """Base class for timeseries model training.

    Subclasses **must** override:
        - ``_build_train_argv(device, distributed)`` — return the task-specific
          training CLI argument list.
        - ``_build_test_argv(device, data_path, model_path, output_dir)`` —
          return the task-specific test CLI argument list.

    Subclasses **must** set the class attributes:
        - ``_train_module`` — reference to the task-specific ``train`` module.
        - ``_test_module`` — reference to the task-specific ``test`` module.

    Subclasses **may** override:
        - ``_get_extra_init_params()`` — return a dict of extra training params
          for the ``__init__`` ``params.update()`` call (default: empty dict).
        - ``_get_quant_min_epochs()`` — minimum epochs for quantization
          retraining (default: 10).
        - ``_get_quant_extra_argv()`` — extra argv entries for quantization
          retraining (default: empty list).
    """

    _train_module = None
    _test_module = None

    @classmethod
    def init_params(cls, *args, **kwargs):
        params = dict(training=dict())
        params = utils.ConfigDict(params, *args, **kwargs)
        return params

    def __init__(self, *args, quit_event=None, **kwargs):
        self.params = self.init_params(*args, **kwargs)
        self.quit_event = quit_event

        log_summary_regex = {
            'js': [
                # Floating Point Training Metrics
                {'type': 'Epoch (FloatTrain)', 'name': 'Epoch (FloatTrain)', 'description': 'Epochs (FloatTrain)',
                 'unit': 'Epoch', 'value': None,
                 'regex': [
                     {'op': 'search', 'pattern': r'FloatTrain: Epoch:\s+\[(?<eid>\d+)\]\s+Total', 'groupId': 'eid'}],
                 },
                {'type': 'Training Loss (FloatTrain)', 'name': 'Loss (FloatTrain)',
                 'description': 'Training Loss (FloatTrain)', 'unit': 'Loss', 'value': None,
                 'regex': [{'op': 'search',
                            'pattern': r'FloatTrain: Epoch:\s\[\d+\]\sTotal\stime:[\s\S]*?loss\:\s+(?<loss>\d+\.\d+)',
                            'groupId': 'loss'}],
                 },
                {'type': 'Validation Accuracy (FloatTrain)', 'name': 'Accuracy (FloatTrain)',
                 'description': 'Validation Accuracy (FloatTrain)', 'unit': 'Accuracy Top-1%', 'value': None,
                 'regex': [{'op': 'search', 'pattern': r'FloatTrain: Test:\s+\s+Acc@1\s+(?<accuracy>[-+e\d+\.\d+]+)',
                            'groupId': 'accuracy', 'scale_factor': 1}],
                 },
                {'type': 'F1-Score (FloatTrain)', 'name': 'F1-Score (FloatTrain)',
                 'description': 'F1-Score (FloatTrain)', 'unit': 'F1-Score', 'value': None,
                 'regex': [{'op': 'search', 'pattern': r'FloatTrain: Test:\s+\s+F1-Score\s+(?<f1score>[-+e\d+\.\d+]+)',
                            'groupId': 'f1score', 'scale_factor': 1}],
                 },
                {'type': 'Confusion Matrix (FloatTrain)', 'name': 'Confusion Matrix (FloatTrain)',
                 'description': 'Confusion Matrix (FloatTrain)', 'unit': 'Confusion Matrix', 'value': None,
                 'regex': [{'op': 'search',
                            'pattern': r'FloatTrain:\s*Confusion Matrix:(\r\n|\r|\n)(?<cm>[\w\s\:\=\+\-\|]+)\s*INFO',
                            'groupId': 'cm', 'scale_factor': 1}],
                 },
                # Quantized Training
                {'type': 'Epoch (QuantTrain)', 'name': 'Epoch (QuantTrain)', 'description': 'Epochs (QuantTrain)',
                 'unit': 'Epoch', 'value': None,
                 'regex': [
                     {'op': 'search', 'pattern': r'QuantTrain: Epoch:\s+\[(?<eid>\d+)\]\s+Total', 'groupId': 'eid'}],
                 },
                {'type': 'Training Loss (QuantTrain)', 'name': 'Loss (QuantTrain)',
                 'description': 'Training Loss (QuantTrain)', 'unit': 'Loss', 'value': None,
                 'regex': [{'op': 'search',
                            'pattern': r'QuantTrain: Epoch:\s\[\d+\]\sTotal\stime:[\s\S]*?loss\:\s+(?<loss>\d+\.\d+)',
                            'groupId': 'loss'}],
                 },
                {'type': 'F1-Score (QuantTrain)', 'name': 'F1-Score (QuantTrain)',
                 'description': 'F1-Score (QuantTrain)', 'unit': 'F1-Score', 'value': None,
                 'regex': [{'op': 'search', 'pattern': r'QuantTrain: Test:\s+\s+F1-Score\s+(?<f1score>[-+e\d+\.\d+]+)',
                            'groupId': 'f1score', 'scale_factor': 1}],
                 },
                {'type': 'Confusion Matrix (QuantTrain)', 'name': 'Confusion Matrix (QuantTrain)',
                 'description': 'Confusion Matrix (QuantTrain)', 'unit': 'Confusion Matrix', 'value': None,
                 'regex': [{'op': 'search',
                            'pattern': r'QuantTrain:\s*Confusion Matrix:(\r\n|\r|\n)(?<cm>[\w\s\:\=\+\-\|]+)\s*INFO',
                            'groupId': 'cm', 'scale_factor': 1}],
                 },
                {'type': 'Validation Accuracy (QuantTrain)', 'name': 'Accuracy (QuantTrain)',
                 'description': 'Validation Accuracy (QuantTrain)', 'unit': 'Accuracy Top-1%', 'value': None,
                 'regex': [{'op': 'search', 'pattern': r'QuantTrain: Test:\s+\s+Acc@1\s+(?<accuracy>[-+e\d+\.\d+]+)',
                            'groupId': 'accuracy', 'scale_factor': 1}],
                 },
                # Best Epoch QuantTrain Metrics
                {'type': 'Epoch (QuantTrain, BestEpoch)', 'name': 'Epoch (QuantTrain, BestEpoch)', 'description': 'Epochs (QuantTrain, BestEpoch)',
                 'unit': 'Epoch', 'value': None,
                 'regex': [
                     {'op': 'search', 'pattern': r'QuantTrain.BestEpoch: Best Epoch:\s+(?<eid>\d+)\s+', 'groupId': 'eid'}],
                 },
                {'type': 'F1-Score (QuantTrain, BestEpoch)', 'name': 'F1-Score (QuantTrain, BestEpoch)',
                 'description': 'F1-Score (QuantTrain, BestEpoch)', 'unit': 'F1-Score', 'value': None,
                 'regex': [{'op': 'search', 'pattern': r'QuantTrain.BestEpoch:\s+F1-Score\s+(?<f1score>[-+e\d+\.\d+]+)',
                            'groupId': 'f1score', 'scale_factor': 1}],
                 },
                {'type': 'Confusion Matrix (QuantTrain, BestEpoch)', 'name': 'Confusion Matrix (QuantTrain, BestEpoch)',
                 'description': 'Confusion Matrix (QuantTrain, BestEpoch)', 'unit': 'Confusion Matrix', 'value': None,
                 'regex': [{'op': 'search',
                            'pattern': r'QuantTrain.BestEpoch:\s*Confusion Matrix:(\r\n|\r|\n)(?<cm>[\w\s\:\=\+\-\|]+)\s*INFO',
                            'groupId': 'cm', 'scale_factor': 1}],
                 },
                {'type': 'Validation Accuracy (QuantTrain, BestEpoch)', 'name': 'Accuracy (QuantTrain, BestEpoch)',
                 'description': 'Validation Accuracy (QuantTrain, BestEpoch)', 'unit': 'Accuracy Top-1%', 'value': None,
                 'regex': [{'op': 'search', 'pattern': r'QuantTrain.BestEpoch: Acc@1\s+(?<accuracy>[-+e\d+\.\d+]+)',
                            'groupId': 'accuracy', 'scale_factor': 1}],
                 },
                # Best Epoch FloatTrain Metrics
                {'type': 'Epoch (FloatTrain, BestEpoch)', 'name': 'Epoch (FloatTrain, BestEpoch)',
                 'description': 'Epochs (FloatTrain, BestEpoch)',
                 'unit': 'Epoch', 'value': None,
                 'regex': [
                     {'op': 'search', 'pattern': r'FloatTrain.BestEpoch\s*: Best Epoch:\s+(?<eid>\d+)\s+',
                      'groupId': 'eid'}],
                 },
                {'type': 'F1-Score (FloatTrain, BestEpoch)', 'name': 'F1-Score (FloatTrain, BestEpoch)',
                 'description': 'F1-Score (FloatTrain, BestEpoch)', 'unit': 'F1-Score', 'value': None,
                 'regex': [
                     {'op': 'search', 'pattern': r'FloatTrain.BestEpoch\s*:\s+F1-Score\s+(?<f1score>[-+e\d+\.\d+]+)',
                      'groupId': 'f1score', 'scale_factor': 1}],
                 },
                {'type': 'Confusion Matrix (FloatTrain, BestEpoch)', 'name': 'Confusion Matrix (FloatTrain, BestEpoch)',
                 'description': 'Confusion Matrix (FloatTrain, BestEpoch)', 'unit': 'Confusion Matrix', 'value': None,
                 'regex': [{'op': 'search',
                            'pattern': r'FloatTrain.BestEpoch\s*:\s*Confusion Matrix:(\r\n|\r|\n)(?<cm>[\w\s\:\=\+\-\|]+)\s*INFO',
                            'groupId': 'cm', 'scale_factor': 1}],
                 },
                {'type': 'Validation Accuracy (FloatTrain, BestEpoch)', 'name': 'Accuracy (FloatTrain, BestEpoch)',
                 'description': 'Validation Accuracy (FloatTrain, BestEpoch)', 'unit': 'Accuracy Top-1%', 'value': None,
                 'regex': [{'op': 'search', 'pattern': r'FloatTrain.BestEpoch\s*: Acc@1\s+(?<accuracy>[-+e\d+\.\d+]+)',
                            'groupId': 'accuracy', 'scale_factor': 1}],
                 },
                # Test data
                {'type': 'Test Accuracy (Test Data)', 'name': 'Accuracy (Test Data)',
                 'description': 'Test Accuracy (Test Data)', 'unit': 'Accuracy Top-1%', 'value': None,
                 'regex': [{'op': 'search', 'pattern': r'test_data\s*:\s*Test Data Evaluation Accuracy:\s+(?<accuracy>[-+e\d+\.\d+]+)%',
                            'groupId': 'accuracy', 'scale_factor': 1}],
                 },
                {'type': 'Confusion Matrix (Test Data)', 'name': 'Confusion Matrix',
                 'description': 'Confusion Matrix (Test Data)', 'unit': 'Confusion Matrix', 'value': None,
                 'regex': [{'op': 'search',
                            'pattern': r'test_data\s*:\s*Confusion Matrix:(\r\n|\r|\n)(?<cm>[\w\s\:\=\+\-\|]+)(\r\n|\r|\n)',
                            'groupId': 'cm', 'scale_factor': 1}],
                 },
                {'type': 'Matrix Label', 'name': 'Matrix Label', 'description': 'Matrix Label',
                 'unit': 'Matrix Label', 'value': None,
                    "regex": [{'op': 'search', 'pattern': r'Ground Truth:\s*(?<label>\w+)\s*\|\s*',
                               'scale_factor': 1, 'groupId': 'label'}],
                },
                {'type': 'Matrix Cell', 'name': 'Matrix Cell', 'description': 'Matrix Cell',
                 'unit': 'Matrix Cell', 'value': None,
                 "regex": [{'op': 'search', 'pattern': r'\|\s*(?<cell>\d+)',
                            'scale_factor': 1, 'groupId': 'cell'}],
                 },
            ]
        }

        # update params that are specific to this backend and model
        training_params = dict(
            log_file_path=os.path.join(
                self.params.training.train_output_path if self.params.training.train_output_path else self.params.training.training_path,
                'run.log'),
            log_summary_regex=log_summary_regex,
            summary_file_path=os.path.join(self.params.training.training_path, 'summary.yaml'),
            model_checkpoint_path=os.path.join(self.params.training.training_path, 'checkpoint.pth'),
            model_export_path=os.path.join(self.params.training.training_path, 'model.onnx'),
            model_proto_path=None,
            tspa_license_path=os.path.abspath(os.path.join(
                os.path.dirname(tinyml_modelmaker.ai_modules.timeseries.training.tinyml_tinyverse.__file__),
                'LICENSE.txt'))
        )
        # allow subclasses to add extra init params (e.g. classification adds file_level_classification_log_path)
        training_params.update(self._get_extra_init_params())
        self.params.update(training=utils.ConfigDict(**training_params))

        if self.params.training.quantization != TinyMLQuantizationVersion.NO_QUANTIZATION:
            self.params.update(
                training=utils.ConfigDict(
                    model_checkpoint_path_quantization=os.path.join(
                        self.params.training.training_path_quantization, 'checkpoint.pth'),
                    model_export_path_quantization=os.path.join(
                        self.params.training.training_path_quantization, 'model.onnx'),
                )
            )

    # ------------------------------------------------------------------
    # Overridable hooks
    # ------------------------------------------------------------------

    def _get_extra_init_params(self):
        """Return extra training params for ``__init__``'s ``params.update()``.

        Override in subclasses that need additional init-time parameters.
        Must return a dict.
        """
        return {}

    def _build_train_argv(self, device, distributed):
        """Build the task-specific training CLI argument list.

        Must be overridden by every subclass.
        """
        raise NotImplementedError

    def _build_test_argv(self, device, data_path, model_path, output_dir):
        """Build the task-specific test CLI argument list.

        Must be overridden by every subclass.
        """
        raise NotImplementedError

    def _get_quant_min_epochs(self):
        """Minimum epochs for quantization retraining.  Default: 10."""
        return 10

    def _get_quant_extra_argv(self):
        """Extra argv entries appended during quantization retraining.

        Override in subclasses that need task-specific quant args
        (e.g. regression adds ``--lambda-reg``).
        """
        return []

    def _post_process_test_args(self, args):
        """Post-process parsed test arguments before running test.

        Override in subclasses that need to set extra attributes on the
        parsed args namespace (e.g. anomaly detection sets ``cache_dataset``
        and ``gpu``).
        """
        return args

    # ------------------------------------------------------------------
    # Lifecycle methods
    # ------------------------------------------------------------------

    def clear(self):
        # clear the training folder
        shutil.rmtree(self.params.training.training_path, ignore_errors=True)

    def run(self, **kwargs):
        ''''
        The actual training function. Move this to a worker process, if this function is called from a GUI.
        '''
        os.makedirs(self.params.training.training_path, exist_ok=True)

        distributed = 1 if self.params.training.num_gpus > 1 else 0
        device = 'cpu'
        if self.params.training.num_gpus > 0:
            if torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cuda'

        # Build task-specific training argv
        argv = self._build_train_argv(device, distributed)

        args = self._train_module.get_args_parser().parse_args(argv)
        args.quit_event = self.quit_event
        if not utils.misc_utils.str2bool(self.params.testing.skip_train):
            if utils.misc_utils.str2bool(self.params.training.run_quant_train_only):
                if self.params.training.quantization != TinyMLQuantizationVersion.NO_QUANTIZATION:
                    argv = argv[:-2]  # Remove --output-dir <output-dir>
                    argv.extend([
                        '--output-dir', f'{self.params.training.training_path_quantization}',
                        '--quantization', f'{self.params.training.quantization}',
                        '--quantization-method', f'{self.params.training.quantization_method}',
                        '--weight-bitwidth', f'{self.params.training.quantization_weight_bitwidth}',
                        '--activation-bitwidth', f'{self.params.training.quantization_activation_bitwidth}',
                    ]),

                    args = self._train_module.get_args_parser().parse_args(argv)
                    args.quit_event = self.quit_event
                    self._train_module.run(args)
                else:
                    raise f"quantization cannot be {TinyMLQuantizationVersion.NO_QUANTIZATION} if run_quant_train_only argument is chosen"
            else:
                self._train_module.run(args)

                if utils.misc_utils.str2bool(self.params.data_processing_feature_extraction.store_feat_ext_data) and utils.misc_utils.str2bool(self.params.data_processing_feature_extraction.dont_train_just_feat_ext):
                    return self.params

                if self.params.training.quantization != TinyMLQuantizationVersion.NO_QUANTIZATION:
                    argv = argv[:-8]
                    # remove --store-feat-ext-data <True/False> --epochs <epochs> --lr <lr> --output-dir <output-dir>
                    argv.extend([
                        '--output-dir', f'{self.params.training.training_path_quantization}',
                        '--epochs', f'{max(self._get_quant_min_epochs(), self.params.training.training_epochs // 5)}',
                        '--lr', f'{self.params.training.learning_rate / 100}',
                    ])
                    argv.extend(self._get_quant_extra_argv())
                    argv.extend([
                        '--weights', f'{self.params.training.model_checkpoint_path}',
                        '--quantization', f'{self.params.training.quantization}',
                        '--quantization-method', f'{self.params.training.quantization_method}',
                        '--weight-bitwidth', f'{self.params.training.quantization_weight_bitwidth}',
                        '--activation-bitwidth', f'{self.params.training.quantization_activation_bitwidth}',
                        '--lr-warmup-epochs', '0',
                        '--store-feat-ext-data', 'False'
                    ])

                    args = self._train_module.get_args_parser().parse_args(argv)
                    args.quit_event = self.quit_event
                    self._train_module.run(args)

        if utils.misc_utils.str2bool(self.params.testing.enable):
            if self.params.testing.test_data and (os.path.exists(self.params.testing.test_data)):
                data_path = self.params.testing.test_data
            else:
                data_path = os.path.join(self.params.dataset.dataset_path, self.params.dataset.data_dir)

            if self.params.testing.model_path and (os.path.exists(self.params.testing.model_path)):
                model_path = self.params.testing.model_path
            else:
                if self.params.training.quantization == TinyMLQuantizationVersion.NO_QUANTIZATION:
                    model_path = os.path.join(self.params.training.training_path, 'model.onnx')
                    output_dir = self.params.training.training_path
                else:
                    model_path = os.path.join(self.params.training.training_path_quantization, 'model.onnx')
                    output_dir = self.params.training.training_path_quantization

            argv = self._build_test_argv(device, data_path, model_path, output_dir)
            args = self._test_module.get_args_parser().parse_args(argv)
            args.quit_event = self.quit_event
            args = self._post_process_test_args(args)
            self._test_module.run(args)

        return self.params

    def stop(self):
        if self.quit_event is not None:
            self.quit_event.set()
            return True
        #
        return False

    def get_params(self):
        return self.params
