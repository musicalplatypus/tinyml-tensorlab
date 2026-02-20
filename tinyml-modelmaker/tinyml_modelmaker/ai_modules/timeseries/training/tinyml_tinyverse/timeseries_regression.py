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

import os
from copy import deepcopy

from tinyml_tinyverse.references.timeseries_regression import test_onnx as test
from tinyml_tinyverse.references.timeseries_regression import train
from tinyml_torchmodelopt.quantization import TinyMLQuantizationVersion

from ..... import utils
from ... import constants
from .base_training import BaseModelTraining
from .base_training import get_model_descriptions as _base_get_model_descriptions
from .base_training import get_model_description as _base_get_model_description
from .device_run_info import DEVICE_RUN_INFO

this_dir_path = os.path.dirname(os.path.abspath(__file__))
repo_parent_path = os.path.abspath(os.path.join(this_dir_path, '..', '..', '..', '..', '..', '..'))

model_info_str = "Inference time numbers are for comparison purposes only. (Input Size: {})"
template_model_description = dict(
    common=dict(
        task_category=constants.TASK_CATEGORY_TS_REGRESSION,
        task_type=constants.TASK_TYPE_GENERIC_TS_REGRESSION,
        generic_model=True,
        model_details='',
    ),
    download=dict(download_url='', download_path=''),
    training=dict(
        quantization=TinyMLQuantizationVersion.QUANTIZATION_TINPU,
        with_input_batchnorm=True,
        dataset_loader='GenericTSDatasetReg',
        training_backend=constants.TRAINING_BACKEND_TINYML_TINYVERSE,
        model_training_id='',
        model_name='',
        learning_rate=2e-3,
        model_spec=None,
        batch_size=constants.TRAINING_BATCH_SIZE_DEFAULT[constants.TASK_TYPE_GENERIC_TS_REGRESSION],
        target_devices={
                constants.TARGET_DEVICE_F280013: dict(model_selection_factor=None),
                constants.TARGET_DEVICE_F280015: dict(model_selection_factor=None),
                constants.TARGET_DEVICE_F28003: dict(model_selection_factor=None),
                constants.TARGET_DEVICE_F28004: dict(model_selection_factor=None),
                constants.TARGET_DEVICE_F2837: dict(model_selection_factor=None),
                constants.TARGET_DEVICE_F28P65: dict(model_selection_factor=None),
                constants.TARGET_DEVICE_F28P55: dict(model_selection_factor=None),},
        training_devices={constants.TRAINING_DEVICE_CPU: True, constants.TRAINING_DEVICE_CUDA: True, constants.TRAINING_DEVICE_MPS: True,},
    ),
    compilation=dict()
)
_model_descriptions = {
    # Regression Models
    'TimeSeries_Generic_Regr_10k_t': utils.deep_update_dict(deepcopy(template_model_description), {
        'common': dict(model_details='Regression Model with 10k params. 3 Conv+BatchNorm+Relu layers + 2 Linear Layer.'),
        'training': dict(
            model_training_id='REG_TS_GEN_BASE_10K',
            model_name='TimeSeries_Generic_Regr_10k_t',
            target_devices={
                constants.TARGET_DEVICE_F280013: dict(model_selection_factor=None) | (DEVICE_RUN_INFO['TimeSeries_Generic_13k_t'][constants.TARGET_DEVICE_F280013]),
                constants.TARGET_DEVICE_F280015: dict(model_selection_factor=None) | (DEVICE_RUN_INFO['TimeSeries_Generic_13k_t'][constants.TARGET_DEVICE_F280015]),
                constants.TARGET_DEVICE_F28003: dict(model_selection_factor=None) | (DEVICE_RUN_INFO['TimeSeries_Generic_13k_t'][constants.TARGET_DEVICE_F28003]),
                constants.TARGET_DEVICE_F28004: dict(model_selection_factor=None) | (DEVICE_RUN_INFO['TimeSeries_Generic_13k_t'][constants.TARGET_DEVICE_F28004]),
                constants.TARGET_DEVICE_F2837: dict(model_selection_factor=None) | (DEVICE_RUN_INFO['TimeSeries_Generic_13k_t'][constants.TARGET_DEVICE_F2837]),
                constants.TARGET_DEVICE_F28P65: dict(model_selection_factor=None) | (DEVICE_RUN_INFO['TimeSeries_Generic_13k_t'][constants.TARGET_DEVICE_F28P65]),
                constants.TARGET_DEVICE_F28P55: dict(model_selection_factor=None) | (DEVICE_RUN_INFO['TimeSeries_Generic_13k_t'][constants.TARGET_DEVICE_F28P55]),
            },
        ),
    }),
    'TimeSeries_Generic_Regr_3k_t': utils.deep_update_dict(deepcopy(template_model_description), {
        'common': dict(model_details='Regression Model with 3k params. 4 layers of MLPs (Fully Connected Layers)'),
        'training': dict(
            model_training_id='REG_TS_GEN_BASE_3K',
            model_name='TimeSeries_Generic_Regr_3k_t',
            target_devices={
                constants.TARGET_DEVICE_F280013: dict(model_selection_factor=None) | (DEVICE_RUN_INFO['TimeSeries_Generic_13k_t'][constants.TARGET_DEVICE_F280013]),
                constants.TARGET_DEVICE_F280015: dict(model_selection_factor=None) | (DEVICE_RUN_INFO['TimeSeries_Generic_13k_t'][constants.TARGET_DEVICE_F280015]),
                constants.TARGET_DEVICE_F28003: dict(model_selection_factor=None) | (DEVICE_RUN_INFO['TimeSeries_Generic_13k_t'][constants.TARGET_DEVICE_F28003]),
                constants.TARGET_DEVICE_F28004: dict(model_selection_factor=None) | (DEVICE_RUN_INFO['TimeSeries_Generic_13k_t'][constants.TARGET_DEVICE_F28004]),
                constants.TARGET_DEVICE_F2837: dict(model_selection_factor=None) | (DEVICE_RUN_INFO['TimeSeries_Generic_13k_t'][constants.TARGET_DEVICE_F2837]),
                constants.TARGET_DEVICE_F28P65: dict(model_selection_factor=None) | (DEVICE_RUN_INFO['TimeSeries_Generic_13k_t'][constants.TARGET_DEVICE_F28P65]),
                constants.TARGET_DEVICE_F28P55: dict(model_selection_factor=None) | (DEVICE_RUN_INFO['TimeSeries_Generic_13k_t'][constants.TARGET_DEVICE_F28P55]),
            },
        ),
    }),
        'Reg_Washing_Machine_13K_t': utils.deep_update_dict(deepcopy(template_model_description), {
        'common': dict(model_details='Regression Model for washing machine with 13K params. 3 Conv+BatchNorm+Relu layers + Linear Layer.'),
        'training': dict(
            model_training_id='REG_TS_CNN_13K',
            model_name='Reg_Washing_Machine_13K_t',
            target_devices={
                constants.TARGET_DEVICE_F280013: dict(model_selection_factor=None) | (DEVICE_RUN_INFO['TimeSeries_Generic_13k_t'][constants.TARGET_DEVICE_F280013]),
                constants.TARGET_DEVICE_F280015: dict(model_selection_factor=None) | (DEVICE_RUN_INFO['TimeSeries_Generic_13k_t'][constants.TARGET_DEVICE_F280015]),
                constants.TARGET_DEVICE_F28003: dict(model_selection_factor=None) | (DEVICE_RUN_INFO['TimeSeries_Generic_13k_t'][constants.TARGET_DEVICE_F28003]),
                constants.TARGET_DEVICE_F28004: dict(model_selection_factor=None) | (DEVICE_RUN_INFO['TimeSeries_Generic_13k_t'][constants.TARGET_DEVICE_F28004]),
                constants.TARGET_DEVICE_F2837: dict(model_selection_factor=None) | (DEVICE_RUN_INFO['TimeSeries_Generic_13k_t'][constants.TARGET_DEVICE_F2837]),
                constants.TARGET_DEVICE_F28P65: dict(model_selection_factor=None) | (DEVICE_RUN_INFO['TimeSeries_Generic_13k_t'][constants.TARGET_DEVICE_F28P65]),
                constants.TARGET_DEVICE_F28P55: dict(model_selection_factor=None) | (DEVICE_RUN_INFO['TimeSeries_Generic_13k_t'][constants.TARGET_DEVICE_F28P55]),
            },
        ),
    }),
        'Reg_Washing_Machine_4K_t': utils.deep_update_dict(deepcopy(template_model_description), {
        'common': dict(model_details='Regression Model for washing machine with 4K params. 2 Conv+BatchNorm+Relu layers + Linear Layer.'),
        'training': dict(
            model_training_id='REG_TS_CNN_4K',
            model_name='Reg_Washing_Machine_4K_t',
            target_devices={
                constants.TARGET_DEVICE_F280013: dict(model_selection_factor=None) | (DEVICE_RUN_INFO['TimeSeries_Generic_13k_t'][constants.TARGET_DEVICE_F280013]),
                constants.TARGET_DEVICE_F280015: dict(model_selection_factor=None) | (DEVICE_RUN_INFO['TimeSeries_Generic_13k_t'][constants.TARGET_DEVICE_F280015]),
                constants.TARGET_DEVICE_F28003: dict(model_selection_factor=None) | (DEVICE_RUN_INFO['TimeSeries_Generic_13k_t'][constants.TARGET_DEVICE_F28003]),
                constants.TARGET_DEVICE_F28004: dict(model_selection_factor=None) | (DEVICE_RUN_INFO['TimeSeries_Generic_13k_t'][constants.TARGET_DEVICE_F28004]),
                constants.TARGET_DEVICE_F2837: dict(model_selection_factor=None) | (DEVICE_RUN_INFO['TimeSeries_Generic_13k_t'][constants.TARGET_DEVICE_F2837]),
                constants.TARGET_DEVICE_F28P65: dict(model_selection_factor=None) | (DEVICE_RUN_INFO['TimeSeries_Generic_13k_t'][constants.TARGET_DEVICE_F28P65]),
                constants.TARGET_DEVICE_F28P55: dict(model_selection_factor=None) | (DEVICE_RUN_INFO['TimeSeries_Generic_13k_t'][constants.TARGET_DEVICE_F28P55]),
            },
        ),
    }),
}

enabled_models_list = [
    # Regression Models
    'TimeSeries_Generic_Regr_10k_t',
    'TimeSeries_Generic_Regr_3k_t',
    'Reg_Washing_Machine_13K_t',
    'Reg_Washing_Machine_4K_t'
]


def get_model_descriptions(task_type=None):
    return _base_get_model_descriptions(_model_descriptions, enabled_models_list, task_type)


def get_model_description(model_name):
    return _base_get_model_description(_model_descriptions, enabled_models_list, model_name)


class ModelTraining(BaseModelTraining):
    _train_module = train
    _test_module = test

    def _get_quant_min_epochs(self):
        return 50

    def _get_quant_extra_argv(self):
        return ['--lambda-reg', f'{self.params.training.lambda_reg}']

    def _build_train_argv(self, device, distributed):
        return ['--model', f'{self.params.training.model_training_id}',
                '--dual-op', f'{self.params.training.dual_op}',
                '--model-config', f'{self.params.training.model_config}',
                '--augment-config', f'{self.params.training.augment_config}',
                '--model-spec', f'{self.params.training.model_spec}',
                # '--weights', f'{self.params.training.pretrained_checkpoint_path}',
                '--dataset', 'modelmaker',
                '--dataset-loader', f'{self.params.training.dataset_loader}',
                '--annotation-prefix', f'{self.params.dataset.annotation_prefix}',
                #'--num-classes', f'{self.params.training.num_classes}',
                '--gpus', f'{self.params.training.num_gpus}',
                '--batch-size', f'{self.params.training.batch_size}',
                '--opt', f'{self.params.training.optimizer}',
                '--weight-decay', f'{self.params.training.weight_decay}',
                '--lr-scheduler', f'{self.params.training.lr_scheduler}',
                '--lr-warmup-epochs', '1',
                '--distributed', f'{distributed}',
                '--device', f'{device}',
                # '--out_dir', f'{self.params.training.}',
                '--sampling-rate', f'{self.params.data_processing_feature_extraction.sampling_rate}',
                '--resampling-factor', f'{self.params.data_processing_feature_extraction.resampling_factor}',
                '--new-sr', f'{self.params.data_processing_feature_extraction.new_sr}',
                '--stride-size', f'{self.params.data_processing_feature_extraction.stride_size}',
                '--data-proc-transforms', self.params.data_processing_feature_extraction.data_proc_transforms,
                '--feat-ext-transform', self.params.data_processing_feature_extraction.feat_ext_transform,

                '--generic-model', f'{self.params.common.generic_model}',

                '--feat-ext-store-dir', f'{self.params.data_processing_feature_extraction.feat_ext_store_dir}',
                '--dont-train-just-feat-ext', f'{self.params.data_processing_feature_extraction.dont_train_just_feat_ext}',

                '--frame-size', f'{self.params.data_processing_feature_extraction.frame_size}',
                '--feature-size-per-frame', f'{self.params.data_processing_feature_extraction.feature_size_per_frame}',
                '--num-frame-concat', f'{self.params.data_processing_feature_extraction.num_frame_concat}',
                '--frame-skip', f'{self.params.data_processing_feature_extraction.frame_skip}',
                '--min-bin', f'{self.params.data_processing_feature_extraction.min_bin}',
                '--normalize-bin', f'{self.params.data_processing_feature_extraction.normalize_bin}',
                '--dc-remove', f'{self.params.data_processing_feature_extraction.dc_remove}',
                '--analysis-bandwidth', f'{self.params.data_processing_feature_extraction.analysis_bandwidth}',
                # '--num-channel', f'{self.params.data_processing_feature_extraction.num_channel}',
                '--log-base', f'{self.params.data_processing_feature_extraction.log_base}',
                '--log-mul', f'{self.params.data_processing_feature_extraction.log_mul}',
                '--log-threshold', f'{self.params.data_processing_feature_extraction.log_threshold}',
                '--stacking', f'{self.params.data_processing_feature_extraction.stacking}',
                '--offset', f'{self.params.data_processing_feature_extraction.offset}',
                '--scale', f'{self.params.data_processing_feature_extraction.scale}',
                '--nn-for-feature-extraction', f'{self.params.data_processing_feature_extraction.nn_for_feature_extraction}',
                '--output-dequantize', f'{self.params.training.output_dequantize}',

                #'--tensorboard-logger', 'True',
                '--variables', f'{self.params.data_processing_feature_extraction.variables}',
                '--with-input-batchnorm', f'{self.params.training.with_input_batchnorm}',
                '--lis', f'{self.params.training.log_file_path}',
                # Do not add newer arguments after this line, it will change the behavior of the code.
                '--data-path', os.path.join(self.params.dataset.dataset_path, self.params.dataset.data_dir),
                '--store-feat-ext-data', f'{self.params.data_processing_feature_extraction.store_feat_ext_data}',
                '--epochs', f'{self.params.training.training_epochs}',
                '--lr', f'{self.params.training.learning_rate}',
                '--lambda-reg', f'{self.params.training.lambda_reg}',
                '--output-dir', f'{self.params.training.training_path}',
                ]

    def _build_test_argv(self, device, data_path, model_path, output_dir):
        return [
                '--dataset', 'modelmaker',
                '--dataset-loader', f'{self.params.training.dataset_loader}',
                '--annotation-prefix', f'{self.params.dataset.annotation_prefix}',
                '--gpus', f'{self.params.training.num_gpus}',
                '--batch-size', f'{self.params.training.batch_size}',
                '--distributed', '0',
                '--device', f'{device}',

                '--variables', f'{self.params.data_processing_feature_extraction.variables}',
                '--sampling-rate', f'{self.params.data_processing_feature_extraction.sampling_rate}',
                '--resampling-factor', f'{self.params.data_processing_feature_extraction.resampling_factor}',
                '--new-sr', f'{self.params.data_processing_feature_extraction.new_sr}',
                '--stride-size', f'{self.params.data_processing_feature_extraction.stride_size}',

                '--data-proc-transforms', self.params.data_processing_feature_extraction.data_proc_transforms,
                '--feat-ext-transform', self.params.data_processing_feature_extraction.feat_ext_transform,
                # Arc Fault and Motor Fault Related Params
                '--frame-size', f'{self.params.data_processing_feature_extraction.frame_size}',
                '--feature-size-per-frame', f'{self.params.data_processing_feature_extraction.feature_size_per_frame}',
                '--num-frame-concat', f'{self.params.data_processing_feature_extraction.num_frame_concat}',
                '--frame-skip', f'{self.params.data_processing_feature_extraction.frame_skip}',
                '--min-bin', f'{self.params.data_processing_feature_extraction.min_bin}',
                '--normalize-bin', f'{self.params.data_processing_feature_extraction.normalize_bin}',
                '--dc-remove', f'{self.params.data_processing_feature_extraction.dc_remove}',
                '--analysis-bandwidth', f'{self.params.data_processing_feature_extraction.analysis_bandwidth}',
                # '--num-channel', f'{self.params.data_processing_feature_extraction.num_channel}',
                '--log-base', f'{self.params.data_processing_feature_extraction.log_base}',
                '--log-mul', f'{self.params.data_processing_feature_extraction.log_mul}',
                '--log-threshold', f'{self.params.data_processing_feature_extraction.log_threshold}',
                '--stacking', f'{self.params.data_processing_feature_extraction.stacking}',
                '--offset', f'{self.params.data_processing_feature_extraction.offset}',
                '--scale', f'{self.params.data_processing_feature_extraction.scale}',
                '--nn-for-feature-extraction', f'{self.params.data_processing_feature_extraction.nn_for_feature_extraction}',
                '--output-dequantize', f'{self.params.training.output_dequantize}',

                # '--tensorboard-logger', 'True',
                '--lis', f'{self.params.training.log_file_path}',
                '--data-path', f'{data_path}',
                '--output-dir', output_dir,
                '--model-path', f'{model_path}',
                '--generic-model', f'{self.params.common.generic_model}',
                ]
