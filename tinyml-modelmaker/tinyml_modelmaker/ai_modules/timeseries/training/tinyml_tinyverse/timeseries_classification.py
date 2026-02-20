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

from tinyml_tinyverse.references.timeseries_classification import test_onnx as test
from tinyml_tinyverse.references.timeseries_classification import train
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
template_gui_model_properties = [
    dict(type="group", dynamic=False, name="train_group", label="Training Parameters", default=["training_epochs", "learning_rate"]),
    dict(label="Epochs", name="training_epochs", type="integer", default=50, min=1, max=1000),
    dict(label="Learning Rate", name="learning_rate", type="float", default=0.04, min=0.001, max=0.1, decimal_places=3, increment=0.001)]
template_model_description = dict(
    common=dict(
        task_category=constants.TASK_CATEGORY_TS_CLASSIFICATION,
        task_type=constants.TASK_TYPE_GENERIC_TS_CLASSIFICATION,
        generic_model=True,
        model_details='',
    ),
    download=dict(download_url='', download_path=''),
    training=dict(
        quantization=TinyMLQuantizationVersion.QUANTIZATION_TINPU,
        with_input_batchnorm=True,
        training_backend=constants.TRAINING_BACKEND_TINYML_TINYVERSE,
        model_training_id='',
        model_name='',
        learning_rate=2e-3,
        model_spec=None,
        batch_size=constants.TRAINING_BATCH_SIZE_DEFAULT[constants.TASK_TYPE_GENERIC_TS_CLASSIFICATION],
        target_devices={},
        training_devices={constants.TRAINING_DEVICE_CPU: True, constants.TRAINING_DEVICE_CUDA: True, constants.TRAINING_DEVICE_MPS: True,},
    ),
    compilation=dict()
)
_model_descriptions = {
    'Res_Add_TimeSeries_Generic_3k_t': utils.deep_update_dict(deepcopy(template_model_description), {
		'common': dict(
            model_details='Classification Model with 3k params.\nResidual Connection.\nAdds the branches.\n4 Conv+BatchNorm+Relu layers + Linear Layer.', 
            help_url="file://models/Res_Add_TimeSeries_Generic_3k_t/Res_Add_TimeSeries_Generic_3k_t.md"
        ),
        'training': dict(
            model_training_id='RES_ADD_CNN_TS_GEN_BASE_3K',
            model_name='Res_Add_TimeSeries_Generic_3k_t',
            target_devices={
                constants.TARGET_DEVICE_F280015: dict(model_selection_factor=6) | (DEVICE_RUN_INFO['Res_Add_TimeSeries_Generic_3k_t'][constants.TARGET_DEVICE_F280015]),
                constants.TARGET_DEVICE_F280013: dict(model_selection_factor=6) | (DEVICE_RUN_INFO['Res_Add_TimeSeries_Generic_3k_t'][constants.TARGET_DEVICE_F280013]),
                constants.TARGET_DEVICE_F28003: dict(model_selection_factor=6) | (DEVICE_RUN_INFO['Res_Add_TimeSeries_Generic_3k_t'][constants.TARGET_DEVICE_F28003]),
                constants.TARGET_DEVICE_F28004: dict(model_selection_factor=6) | (DEVICE_RUN_INFO['Res_Add_TimeSeries_Generic_3k_t'][constants.TARGET_DEVICE_F28004]),
                constants.TARGET_DEVICE_F2837: dict(model_selection_factor=6) | (DEVICE_RUN_INFO['Res_Add_TimeSeries_Generic_3k_t'][constants.TARGET_DEVICE_F2837]),
                constants.TARGET_DEVICE_F28P65: dict(model_selection_factor=6) | (DEVICE_RUN_INFO['Res_Add_TimeSeries_Generic_3k_t'][constants.TARGET_DEVICE_F28P65]),
                constants.TARGET_DEVICE_F28P55: dict(model_selection_factor=6) | (DEVICE_RUN_INFO['Res_Add_TimeSeries_Generic_3k_t'][constants.TARGET_DEVICE_F28P55]),
                constants.TARGET_DEVICE_MSPM0G3507: dict(model_selection_factor=6) | (DEVICE_RUN_INFO['Res_Add_TimeSeries_Generic_3k_t'][constants.TARGET_DEVICE_MSPM0G3507]),
                constants.TARGET_DEVICE_MSPM0G5187: dict(model_selection_factor=6) | (DEVICE_RUN_INFO['Res_Add_TimeSeries_Generic_3k_t'][constants.TARGET_DEVICE_MSPM0G5187]),
                constants.TARGET_DEVICE_CC2755: dict(model_selection_factor=6) | (DEVICE_RUN_INFO['Res_Add_TimeSeries_Generic_3k_t'][constants.TARGET_DEVICE_CC2755]),
            },
            properties=[dict(type="group", dynamic=True, script="generictimeseries.py", name="preprocessing_group", label="Preprocessing Parameters", default=[])] + template_gui_model_properties
        ),
    }),
    'Res_Cat_TimeSeries_Generic_3k_t': utils.deep_update_dict(deepcopy(template_model_description), {
		'common': dict(
            model_details='Classification Model with 3k params.\nResidual Connection.\nConcatenates the branches.\n4 Conv+BatchNorm+Relu layers + Linear Layer.', 
            help_url="file://models/Res_Cat_TimeSeries_Generic_3k_t/Res_Cat_TimeSeries_Generic_3k_t.md"
        ),
        'training': dict(
            model_training_id='RES_CAT_CNN_TS_GEN_BASE_3K',
            model_name='Res_Cat_TimeSeries_Generic_3k_t',
            target_devices={
                constants.TARGET_DEVICE_F280013: dict(model_selection_factor=1) | (DEVICE_RUN_INFO['Res_Cat_TimeSeries_Generic_3k_t'][constants.TARGET_DEVICE_F280013]),
                constants.TARGET_DEVICE_F280015: dict(model_selection_factor=1) | (DEVICE_RUN_INFO['Res_Cat_TimeSeries_Generic_3k_t'][constants.TARGET_DEVICE_F280015]),
                constants.TARGET_DEVICE_F28003: dict(model_selection_factor=1) | (DEVICE_RUN_INFO['Res_Cat_TimeSeries_Generic_3k_t'][constants.TARGET_DEVICE_F28003]),
                constants.TARGET_DEVICE_F28004: dict(model_selection_factor=1) | (DEVICE_RUN_INFO['Res_Cat_TimeSeries_Generic_3k_t'][constants.TARGET_DEVICE_F28004]),
                constants.TARGET_DEVICE_F2837: dict(model_selection_factor=1) | (DEVICE_RUN_INFO['Res_Cat_TimeSeries_Generic_3k_t'][constants.TARGET_DEVICE_F2837]),
                constants.TARGET_DEVICE_F28P65: dict(model_selection_factor=1) | (DEVICE_RUN_INFO['Res_Cat_TimeSeries_Generic_3k_t'][constants.TARGET_DEVICE_F28P65]),
                constants.TARGET_DEVICE_F28P55: dict(model_selection_factor=1) | (DEVICE_RUN_INFO['Res_Cat_TimeSeries_Generic_3k_t'][constants.TARGET_DEVICE_F28P55]),
                constants.TARGET_DEVICE_MSPM0G3507: dict(model_selection_factor=1, ) | (DEVICE_RUN_INFO['Res_Cat_TimeSeries_Generic_3k_t'][constants.TARGET_DEVICE_MSPM0G3507]),
                constants.TARGET_DEVICE_MSPM0G5187: dict(model_selection_factor=1) | (DEVICE_RUN_INFO['Res_Cat_TimeSeries_Generic_3k_t'][constants.TARGET_DEVICE_MSPM0G5187]),
                constants.TARGET_DEVICE_CC2755: dict(model_selection_factor=1) | (DEVICE_RUN_INFO['Res_Cat_TimeSeries_Generic_3k_t'][constants.TARGET_DEVICE_CC2755]),
            },
            properties=[dict(type="group", dynamic=True, script="generictimeseries.py", name="preprocessing_group", label="Preprocessing Parameters", default=[])] + template_gui_model_properties
        ),
    }),
    'TimeSeries_Generic_13k_t': utils.deep_update_dict(deepcopy(template_model_description), {
		'common': dict(
            model_details='Classification Model with 13k params. \n6 Conv+BatchNorm+Relu layers + Linear Layer.', 
            help_url="file://models/TimeSeries_Generic_13k_t/TimeSeries_Generic_13k_t.md"
        ),
        'training': dict(
            model_training_id='CNN_TS_GEN_BASE_13K',
            model_name='TimeSeries_Generic_13k_t',
            target_devices={
                constants.TARGET_DEVICE_F280013: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['TimeSeries_Generic_13k_t'][constants.TARGET_DEVICE_F280013]),
                constants.TARGET_DEVICE_F280015: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['TimeSeries_Generic_13k_t'][constants.TARGET_DEVICE_F280015]),
                constants.TARGET_DEVICE_F28003: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['TimeSeries_Generic_13k_t'][constants.TARGET_DEVICE_F28003]),
                constants.TARGET_DEVICE_F28004: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['TimeSeries_Generic_13k_t'][constants.TARGET_DEVICE_F28004]),
                constants.TARGET_DEVICE_F2837: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['TimeSeries_Generic_13k_t'][constants.TARGET_DEVICE_F2837]),
                constants.TARGET_DEVICE_F28P65: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['TimeSeries_Generic_13k_t'][constants.TARGET_DEVICE_F28P65]),
                constants.TARGET_DEVICE_F28P55: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['TimeSeries_Generic_13k_t'][constants.TARGET_DEVICE_F28P55]),
                constants.TARGET_DEVICE_MSPM0G3507: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['TimeSeries_Generic_13k_t'][constants.TARGET_DEVICE_MSPM0G3507]),
                constants.TARGET_DEVICE_MSPM0G5187: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['TimeSeries_Generic_13k_t'][constants.TARGET_DEVICE_MSPM0G5187]),
                constants.TARGET_DEVICE_CC2755: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['TimeSeries_Generic_13k_t'][constants.TARGET_DEVICE_CC2755]),

            },
            properties=[dict(type="group", dynamic=True, script="generictimeseries.py", name="preprocessing_group", label="Preprocessing Parameters", default=[])] + template_gui_model_properties
        ),
    }),
    'TimeSeries_Generic_6k_t': utils.deep_update_dict(deepcopy(template_model_description), {
		'common': dict(
            model_details='Classification Model with 6k params. \n6 Conv+BatchNorm+Relu layers + Linear Layer.\nLean model', 
            help_url="file://models/TimeSeries_Generic_6k_t/TimeSeries_Generic_6k_t.md"
        ),
        'training': dict(
            model_training_id='CNN_TS_GEN_BASE_6K',
            model_name='TimeSeries_Generic_6k_t',
            target_devices={
                constants.TARGET_DEVICE_F280013: dict(model_selection_factor=3) | (DEVICE_RUN_INFO['TimeSeries_Generic_6k_t'][constants.TARGET_DEVICE_F280013]),
                constants.TARGET_DEVICE_F280015: dict(model_selection_factor=3) | (DEVICE_RUN_INFO['TimeSeries_Generic_6k_t'][constants.TARGET_DEVICE_F280015]),
                constants.TARGET_DEVICE_F28003: dict(model_selection_factor=3) | (DEVICE_RUN_INFO['TimeSeries_Generic_6k_t'][constants.TARGET_DEVICE_F28003]),
                constants.TARGET_DEVICE_F28004: dict(model_selection_factor=3) | (DEVICE_RUN_INFO['TimeSeries_Generic_6k_t'][constants.TARGET_DEVICE_F28004]),
                constants.TARGET_DEVICE_F2837: dict(model_selection_factor=3) | (DEVICE_RUN_INFO['TimeSeries_Generic_6k_t'][constants.TARGET_DEVICE_F2837]),
                constants.TARGET_DEVICE_F28P65: dict(model_selection_factor=3) | (DEVICE_RUN_INFO['TimeSeries_Generic_6k_t'][constants.TARGET_DEVICE_F28P65]),
                constants.TARGET_DEVICE_F28P55: dict(model_selection_factor=3) | (DEVICE_RUN_INFO['TimeSeries_Generic_6k_t'][constants.TARGET_DEVICE_F28P55]),
                constants.TARGET_DEVICE_MSPM0G3507: dict(model_selection_factor=3) | (DEVICE_RUN_INFO['TimeSeries_Generic_6k_t'][constants.TARGET_DEVICE_MSPM0G3507]),
                constants.TARGET_DEVICE_MSPM0G5187: dict(model_selection_factor=3) | (DEVICE_RUN_INFO['TimeSeries_Generic_6k_t'][constants.TARGET_DEVICE_MSPM0G5187]),
                constants.TARGET_DEVICE_CC2755: dict(model_selection_factor=3) | (DEVICE_RUN_INFO['TimeSeries_Generic_6k_t'][constants.TARGET_DEVICE_CC2755]),
            },
            properties=[dict(type="group", dynamic=True, script="generictimeseries.py", name="preprocessing_group", label="Preprocessing Parameters", default=[])] + template_gui_model_properties
        ),
    }),
    'TimeSeries_Generic_4k_t': utils.deep_update_dict(deepcopy(template_model_description), {
		'common': dict(model_details='Classification Model with 4k params.\n3 Conv+BatchNorm+Relu layers + Linear Layer.', help_url="file://models/TimeSeries_Generic_4k_t/TimeSeries_Generic_4k_t.md"),
        'training': dict(
            model_training_id='CNN_TS_GEN_BASE_4K',
            model_name='TimeSeries_Generic_4k_t',
            target_devices={
                constants.TARGET_DEVICE_F280013: dict(model_selection_factor=5) | (DEVICE_RUN_INFO['TimeSeries_Generic_4k_t'][constants.TARGET_DEVICE_F280013]),
                constants.TARGET_DEVICE_F280015: dict(model_selection_factor=5) | (DEVICE_RUN_INFO['TimeSeries_Generic_4k_t'][constants.TARGET_DEVICE_F280015]),
                constants.TARGET_DEVICE_F28003: dict(model_selection_factor=5) | (DEVICE_RUN_INFO['TimeSeries_Generic_4k_t'][constants.TARGET_DEVICE_F28003]),
                constants.TARGET_DEVICE_F28004: dict(model_selection_factor=5) | (DEVICE_RUN_INFO['TimeSeries_Generic_4k_t'][constants.TARGET_DEVICE_F28004]),
                constants.TARGET_DEVICE_F2837: dict(model_selection_factor=5) | (DEVICE_RUN_INFO['TimeSeries_Generic_4k_t'][constants.TARGET_DEVICE_F2837]),
                constants.TARGET_DEVICE_F28P65: dict(model_selection_factor=5) | (DEVICE_RUN_INFO['TimeSeries_Generic_4k_t'][constants.TARGET_DEVICE_F28P65]),
                constants.TARGET_DEVICE_F28P55: dict(model_selection_factor=5) | (DEVICE_RUN_INFO['TimeSeries_Generic_4k_t'][constants.TARGET_DEVICE_F28P55]),
                constants.TARGET_DEVICE_MSPM0G3507: dict(model_selection_factor=5) | (DEVICE_RUN_INFO['TimeSeries_Generic_4k_t'][constants.TARGET_DEVICE_MSPM0G3507]),
                constants.TARGET_DEVICE_MSPM0G5187: dict(model_selection_factor=5) | (DEVICE_RUN_INFO['TimeSeries_Generic_4k_t'][constants.TARGET_DEVICE_MSPM0G5187]),
                constants.TARGET_DEVICE_CC2755: dict(model_selection_factor=5) | (DEVICE_RUN_INFO['TimeSeries_Generic_4k_t'][constants.TARGET_DEVICE_CC2755]),
            },
            properties=[dict(type="group", dynamic=True, script="generictimeseries.py", name="preprocessing_group", label="Preprocessing Parameters", default=[])] + template_gui_model_properties
        ),
    }),
    'TimeSeries_Generic_1k_t': utils.deep_update_dict(deepcopy(template_model_description), {
		'common': dict(
            model_details='Classification Model with 1k params.\n4 Conv+BatchNorm+Relu layers + Linear Layer.\nVery lean model', 
            help_url="file://models/TimeSeries_Generic_1k_t/TimeSeries_Generic_1k_t.md"
        ),
        'training': dict(
            model_training_id='CNN_TS_GEN_BASE_1K',
            model_name='TimeSeries_Generic_1k_t',
            target_devices={
                constants.TARGET_DEVICE_F280013: dict(model_selection_factor=2) | (DEVICE_RUN_INFO['TimeSeries_Generic_1k_t'][constants.TARGET_DEVICE_F280013]),
                constants.TARGET_DEVICE_F280015: dict(model_selection_factor=2) | (DEVICE_RUN_INFO['TimeSeries_Generic_1k_t'][constants.TARGET_DEVICE_F280015]),
                constants.TARGET_DEVICE_F28003: dict(model_selection_factor=2) | (DEVICE_RUN_INFO['TimeSeries_Generic_1k_t'][constants.TARGET_DEVICE_F28003]),
                constants.TARGET_DEVICE_F28004: dict(model_selection_factor=2) | (DEVICE_RUN_INFO['TimeSeries_Generic_1k_t'][constants.TARGET_DEVICE_F28004]),
                constants.TARGET_DEVICE_F2837: dict(model_selection_factor=2) | (DEVICE_RUN_INFO['TimeSeries_Generic_1k_t'][constants.TARGET_DEVICE_F2837]),
                constants.TARGET_DEVICE_F28P65: dict(model_selection_factor=2) | (DEVICE_RUN_INFO['TimeSeries_Generic_1k_t'][constants.TARGET_DEVICE_F28P65]),
                constants.TARGET_DEVICE_F28P55: dict(model_selection_factor=2) | (DEVICE_RUN_INFO['TimeSeries_Generic_1k_t'][constants.TARGET_DEVICE_F28P55]),
                constants.TARGET_DEVICE_MSPM0G3507: dict(model_selection_factor=2) | (DEVICE_RUN_INFO['TimeSeries_Generic_1k_t'][constants.TARGET_DEVICE_MSPM0G3507]),
                constants.TARGET_DEVICE_MSPM0G5187: dict(model_selection_factor=2) | (DEVICE_RUN_INFO['TimeSeries_Generic_1k_t'][constants.TARGET_DEVICE_MSPM0G5187]),
                constants.TARGET_DEVICE_CC2755: dict(model_selection_factor=2) | (DEVICE_RUN_INFO['TimeSeries_Generic_1k_t'][constants.TARGET_DEVICE_CC2755]),
            },
            properties=[dict(type="group", dynamic=True, script="generictimeseries.py", name="preprocessing_group", label="Preprocessing Parameters", default=[])] + template_gui_model_properties
        ),
    }),
    'TimeSeries_Generic_100_t': utils.deep_update_dict(deepcopy(template_model_description), {
		'common': dict(
            model_details='Classification Model with 100 params.\n2 Conv+BatchNorm+Relu layers+ Adapt Avg Pool +Linear Layer.\nUltra lean model', 
            help_url="file://models/TimeSeries_Generic_100_t/TimeSeries_Generic_100_t.md"
        ),
        'training': dict(
            model_training_id='CNN_TS_GEN_BASE_100',
            model_name='TimeSeries_Generic_100_t',
            target_devices={
                constants.TARGET_DEVICE_F280013: dict(model_selection_factor=4) | (DEVICE_RUN_INFO['TimeSeries_Generic_100_t'][constants.TARGET_DEVICE_F280013]),
                constants.TARGET_DEVICE_F280015: dict(model_selection_factor=4) | (DEVICE_RUN_INFO['TimeSeries_Generic_100_t'][constants.TARGET_DEVICE_F280015]),
                constants.TARGET_DEVICE_F28003: dict(model_selection_factor=4) | (DEVICE_RUN_INFO['TimeSeries_Generic_100_t'][constants.TARGET_DEVICE_F28003]),
                constants.TARGET_DEVICE_F28004: dict(model_selection_factor=4) | (DEVICE_RUN_INFO['TimeSeries_Generic_100_t'][constants.TARGET_DEVICE_F28004]),
                constants.TARGET_DEVICE_F2837: dict(model_selection_factor=4) | (DEVICE_RUN_INFO['TimeSeries_Generic_100_t'][constants.TARGET_DEVICE_F2837]),
                constants.TARGET_DEVICE_F28P65: dict(model_selection_factor=4) | (DEVICE_RUN_INFO['TimeSeries_Generic_100_t'][constants.TARGET_DEVICE_F28P65]),
                constants.TARGET_DEVICE_F28P55: dict(model_selection_factor=4) | (DEVICE_RUN_INFO['TimeSeries_Generic_100_t'][constants.TARGET_DEVICE_F28P55]),
                constants.TARGET_DEVICE_MSPM0G3507: dict(model_selection_factor=4) | (DEVICE_RUN_INFO['TimeSeries_Generic_100_t'][constants.TARGET_DEVICE_MSPM0G3507]),
                constants.TARGET_DEVICE_MSPM0G5187: dict(model_selection_factor=4) | (DEVICE_RUN_INFO['TimeSeries_Generic_100_t'][constants.TARGET_DEVICE_MSPM0G5187]),
                constants.TARGET_DEVICE_CC2755: dict(model_selection_factor=4) | (DEVICE_RUN_INFO['TimeSeries_Generic_100_t'][constants.TARGET_DEVICE_CC2755]),
            },
            properties=[dict(type="group", dynamic=True, script="generictimeseries.py", name="preprocessing_group", label="Preprocessing Parameters", default=[])] + template_gui_model_properties
        ),
    }),
    'ArcFault_model_1400_t': utils.deep_update_dict(deepcopy(template_model_description), {
        'common': dict(
            task_type=constants.TASK_TYPE_ARC_FAULT,
            generic_model=False,
            model_details='TI\u2122 handcrafted model.\nClassification Model with 1400+ params.\nMost accurate model, use for complex data scenarios.',
            help_url="file://models/ArcFault_model_1400_t/ArcFault_model_1400_t.md"
        ),
        'training': dict(
            model_training_id='CNN_AF_3L_1400',
            model_name='ArcFault_model_1400_t',
            learning_rate=0.04,
            model_spec=os.path.join(repo_parent_path, 'tinyml-mlbackend', 'tinyml_proprietary_models', 'cnn_af_3l.py'),
            batch_size=constants.TRAINING_BATCH_SIZE_DEFAULT[constants.TASK_TYPE_ARC_FAULT],
            target_devices={
                constants.TARGET_DEVICE_F280013: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['ArcFault_model_1400_t'][constants.TARGET_DEVICE_F280013]),
                constants.TARGET_DEVICE_F280015: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['ArcFault_model_1400_t'][constants.TARGET_DEVICE_F280015]),
                constants.TARGET_DEVICE_F28003: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['ArcFault_model_1400_t'][constants.TARGET_DEVICE_F28003]),
                constants.TARGET_DEVICE_F28004: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['ArcFault_model_1400_t'][constants.TARGET_DEVICE_F28004]),
                constants.TARGET_DEVICE_F2837: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['ArcFault_model_1400_t'][constants.TARGET_DEVICE_F2837]),
                constants.TARGET_DEVICE_F28P65: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['ArcFault_model_1400_t'][constants.TARGET_DEVICE_F28P65]),
                constants.TARGET_DEVICE_F28P55: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['ArcFault_model_1400_t'][constants.TARGET_DEVICE_F28P55]),
                constants.TARGET_DEVICE_MSPM0G3507: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['ArcFault_model_1400_t'][constants.TARGET_DEVICE_MSPM0G3507]),
                constants.TARGET_DEVICE_MSPM0G5187: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['ArcFault_model_1400_t'][constants.TARGET_DEVICE_MSPM0G5187]),
            },
            properties=[dict(type="group", dynamic=True, script="arcfault.py", name="preprocessing_group", label="Preprocessing Parameters", default=[])] + template_gui_model_properties
        ),
    }),
    'ArcFault_model_700_t': utils.deep_update_dict(deepcopy(template_model_description), {
        'common': dict(
            task_type=constants.TASK_TYPE_ARC_FAULT,
            generic_model=False,
            model_details='TI\u2122 handcrafted model.\nClassification Model with 700+ params.\nLarge model, sweet spot between inference speed & memory occupied.',
            help_url="file://models/ArcFault_model_700_t/ArcFault_model_700_t.md"
        ),
        'training': dict(
            model_training_id='CNN_AF_3L_700',
            model_name='ArcFault_model_700_t',
            learning_rate=0.04,
            model_spec=os.path.join(repo_parent_path, 'tinyml-mlbackend', 'tinyml_proprietary_models', 'cnn_af_3l.py'),
            batch_size=constants.TRAINING_BATCH_SIZE_DEFAULT[constants.TASK_TYPE_ARC_FAULT],
            target_devices={
                constants.TARGET_DEVICE_F280013: dict(model_selection_factor=1) | (DEVICE_RUN_INFO['ArcFault_model_700_t'][constants.TARGET_DEVICE_F280013]),
                constants.TARGET_DEVICE_F280015: dict(model_selection_factor=1) | (DEVICE_RUN_INFO['ArcFault_model_700_t'][constants.TARGET_DEVICE_F280015]),
                constants.TARGET_DEVICE_F28003: dict(model_selection_factor=1) | (DEVICE_RUN_INFO['ArcFault_model_700_t'][constants.TARGET_DEVICE_F28003]),
                constants.TARGET_DEVICE_F28004: dict(model_selection_factor=1) | (DEVICE_RUN_INFO['ArcFault_model_700_t'][constants.TARGET_DEVICE_F28004]),
                constants.TARGET_DEVICE_F2837: dict(model_selection_factor=1) | (DEVICE_RUN_INFO['ArcFault_model_700_t'][constants.TARGET_DEVICE_F2837]),
                constants.TARGET_DEVICE_F28P65: dict(model_selection_factor=1) | (DEVICE_RUN_INFO['ArcFault_model_700_t'][constants.TARGET_DEVICE_F28P65]),
                constants.TARGET_DEVICE_F28P55: dict(model_selection_factor=1) | (DEVICE_RUN_INFO['ArcFault_model_700_t'][constants.TARGET_DEVICE_F28P55]),
                constants.TARGET_DEVICE_MSPM0G3507: dict(model_selection_factor=1) | (DEVICE_RUN_INFO['ArcFault_model_700_t'][constants.TARGET_DEVICE_MSPM0G3507]),
                constants.TARGET_DEVICE_MSPM0G5187: dict(model_selection_factor=1) | (DEVICE_RUN_INFO['ArcFault_model_700_t'][constants.TARGET_DEVICE_MSPM0G5187]),

            },
            properties=[dict(type="group", dynamic=True, script="arcfault.py", name="preprocessing_group", label="Preprocessing Parameters", default=[])] + template_gui_model_properties
        ),
    }),
    'ArcFault_model_300_t': utils.deep_update_dict(deepcopy(template_model_description), {
        'common': dict(
            task_type=constants.TASK_TYPE_ARC_FAULT,
            generic_model=False,
            model_details='TI\u2122 handcrafted model.\nClassification Model with 300+ params. Faster than the 700 & 1400 variant, but also handles less complex data.',
            help_url="file://models/ArcFault_model_300_t/ArcFault_model_300_t.md"
        ),
        'training': dict(
            model_training_id='CNN_AF_3L_300',
            model_name='ArcFault_model_300_t',
            learning_rate=0.04,
            model_spec=os.path.join(repo_parent_path, 'tinyml-mlbackend', 'tinyml_proprietary_models', 'cnn_af_3l.py'),
            batch_size=constants.TRAINING_BATCH_SIZE_DEFAULT[constants.TASK_TYPE_ARC_FAULT],
            target_devices={
                constants.TARGET_DEVICE_F280013: dict(model_selection_factor=2) | (DEVICE_RUN_INFO['ArcFault_model_300_t'][constants.TARGET_DEVICE_F280013]),
                constants.TARGET_DEVICE_F280015: dict(model_selection_factor=2) | (DEVICE_RUN_INFO['ArcFault_model_300_t'][constants.TARGET_DEVICE_F280015]),
                constants.TARGET_DEVICE_F28003: dict(model_selection_factor=2) | (DEVICE_RUN_INFO['ArcFault_model_300_t'][constants.TARGET_DEVICE_F28003]),
                constants.TARGET_DEVICE_F28004: dict(model_selection_factor=2) | (DEVICE_RUN_INFO['ArcFault_model_300_t'][constants.TARGET_DEVICE_F28004]),
                constants.TARGET_DEVICE_F2837: dict(model_selection_factor=2) | (DEVICE_RUN_INFO['ArcFault_model_300_t'][constants.TARGET_DEVICE_F2837]),
                constants.TARGET_DEVICE_F28P65: dict(model_selection_factor=2) | (DEVICE_RUN_INFO['ArcFault_model_300_t'][constants.TARGET_DEVICE_F28P65]),
                constants.TARGET_DEVICE_F28P55: dict(model_selection_factor=2) | (DEVICE_RUN_INFO['ArcFault_model_300_t'][constants.TARGET_DEVICE_F28P55]),
                constants.TARGET_DEVICE_MSPM0G3507: dict(model_selection_factor=2) | (DEVICE_RUN_INFO['ArcFault_model_300_t'][constants.TARGET_DEVICE_MSPM0G3507]),
                constants.TARGET_DEVICE_MSPM0G5187: dict(model_selection_factor=2) | (DEVICE_RUN_INFO['ArcFault_model_300_t'][constants.TARGET_DEVICE_MSPM0G5187]),
            },
        properties=[dict(type="group", dynamic=True, script="arcfault.py", name="preprocessing_group", label="Preprocessing Parameters", default=[])] + template_gui_model_properties
        ),
    }),
    'ArcFault_model_200_t': utils.deep_update_dict(deepcopy(template_model_description), {
        'common': dict(
            task_type=constants.TASK_TYPE_ARC_FAULT,
            generic_model=False,
            model_details='TI\u2122 handcrafted model.\nClassification Model with 200+ params.\nSimplest, smallest & fastest model.',
            help_url="file://models/ArcFault_model_200_t/ArcFault_model_200_t.md"
        ),
        'training': dict(
            model_training_id='CNN_AF_3L_200',
            model_name='ArcFault_model_200_t',
            learning_rate=0.04,
            model_spec=os.path.join(repo_parent_path, 'tinyml-mlbackend', 'tinyml_proprietary_models', 'cnn_af_3l.py'),
            batch_size=constants.TRAINING_BATCH_SIZE_DEFAULT[constants.TASK_TYPE_ARC_FAULT],
            target_devices={
                constants.TARGET_DEVICE_F280013: dict(model_selection_factor=3) | (DEVICE_RUN_INFO['ArcFault_model_200_t'][constants.TARGET_DEVICE_F280013]),
                constants.TARGET_DEVICE_F280015: dict(model_selection_factor=3) | (DEVICE_RUN_INFO['ArcFault_model_200_t'][constants.TARGET_DEVICE_F280015]),
                constants.TARGET_DEVICE_F28003: dict(model_selection_factor=3) | (DEVICE_RUN_INFO['ArcFault_model_200_t'][constants.TARGET_DEVICE_F28003]),
                constants.TARGET_DEVICE_F28004: dict(model_selection_factor=3) | (DEVICE_RUN_INFO['ArcFault_model_200_t'][constants.TARGET_DEVICE_F28004]),
                constants.TARGET_DEVICE_F2837: dict(model_selection_factor=3) | (DEVICE_RUN_INFO['ArcFault_model_200_t'][constants.TARGET_DEVICE_F2837]),
                constants.TARGET_DEVICE_F28P65: dict(model_selection_factor=3) | (DEVICE_RUN_INFO['ArcFault_model_200_t'][constants.TARGET_DEVICE_F28P65]),
                constants.TARGET_DEVICE_F28P55: dict(model_selection_factor=3) | (DEVICE_RUN_INFO['ArcFault_model_200_t'][constants.TARGET_DEVICE_F28P55]),
                constants.TARGET_DEVICE_MSPM0G3507: dict(model_selection_factor=3) | (DEVICE_RUN_INFO['ArcFault_model_200_t'][constants.TARGET_DEVICE_MSPM0G3507]),
                constants.TARGET_DEVICE_MSPM0G5187: dict(model_selection_factor=3) | (DEVICE_RUN_INFO['ArcFault_model_200_t'][constants.TARGET_DEVICE_MSPM0G5187]),
            },
            properties=[dict(type="group", dynamic=True, script="arcfault.py", name="preprocessing_group", label="Preprocessing Parameters", default=[])] + template_gui_model_properties
        ),
    }),
    'MotorFault_model_1_t': utils.deep_update_dict(deepcopy(template_model_description), {
        'common': dict(
            task_type=constants.TASK_TYPE_MOTOR_FAULT,
            generic_model=False,
            model_details='TI\u2122 handcrafted model.\nClassification Model with ~0.6k params.\nSimplest model.',
            help_url="file://models/MotorFault_model_1_t/MotorFault_model_1_t.md"
        ),
        'training': dict(
            model_training_id='CNN_MF_1L',
            model_name='MotorFault_model_1_t',
            learning_rate=0.01,
            model_spec=os.path.join(repo_parent_path, 'tinyml-mlbackend', 'tinyml_proprietary_models', 'cnn_mf_1l.py'),
            batch_size=constants.TRAINING_BATCH_SIZE_DEFAULT[constants.TASK_TYPE_MOTOR_FAULT],
            target_devices={
                constants.TARGET_DEVICE_F280013: dict(model_selection_factor=2) | (DEVICE_RUN_INFO['MotorFault_model_1_t'][constants.TARGET_DEVICE_F280013]),
                constants.TARGET_DEVICE_F280015: dict(model_selection_factor=2) | (DEVICE_RUN_INFO['MotorFault_model_1_t'][constants.TARGET_DEVICE_F280015]),
                constants.TARGET_DEVICE_F28003: dict(model_selection_factor=2) | (DEVICE_RUN_INFO['MotorFault_model_1_t'][constants.TARGET_DEVICE_F28003]),
                constants.TARGET_DEVICE_F28004: dict(model_selection_factor=2) | (DEVICE_RUN_INFO['MotorFault_model_1_t'][constants.TARGET_DEVICE_F28004]),
                constants.TARGET_DEVICE_F2837: dict(model_selection_factor=2) | (DEVICE_RUN_INFO['MotorFault_model_1_t'][constants.TARGET_DEVICE_F2837]),
                constants.TARGET_DEVICE_F28P65: dict(model_selection_factor=2) | (DEVICE_RUN_INFO['MotorFault_model_1_t'][constants.TARGET_DEVICE_F28P65]),
                constants.TARGET_DEVICE_F28P55: dict(model_selection_factor=2) | (DEVICE_RUN_INFO['MotorFault_model_1_t'][constants.TARGET_DEVICE_F28P55]),
                constants.TARGET_DEVICE_MSPM0G3507: dict(model_selection_factor=2) | (DEVICE_RUN_INFO['MotorFault_model_1_t'][constants.TARGET_DEVICE_MSPM0G3507]),
                constants.TARGET_DEVICE_MSPM0G5187: dict(model_selection_factor=2) | (DEVICE_RUN_INFO['MotorFault_model_1_t'][constants.TARGET_DEVICE_MSPM0G5187]),
            },
            properties=[dict(type="group", dynamic=True, script="motorfault.py", name="preprocessing_group", label="Preprocessing Parameters", default=[])] + template_gui_model_properties
        ),
    }),
    'MotorFault_model_2_t': utils.deep_update_dict(deepcopy(template_model_description), {
        'common': dict(
            task_type=constants.TASK_TYPE_MOTOR_FAULT,
            generic_model=False,
            model_details='TI\u2122 handcrafted model.\nClassification Model with 3k params.\nBest & largest of the 3 models, hardest to train.',
            help_url="file://models/MotorFault_model_2_t/MotorFault_model_2_t.md"
        ),
        'training': dict(
            model_training_id='CNN_MF_2L',
            model_name='MotorFault_model_2_t',
            learning_rate=0.01,
            model_spec=os.path.join(repo_parent_path, 'tinyml-mlbackend', 'tinyml_proprietary_models', 'cnn_mf_2l.py'),
            batch_size=constants.TRAINING_BATCH_SIZE_DEFAULT[constants.TASK_TYPE_MOTOR_FAULT],
            target_devices={
                constants.TARGET_DEVICE_F280013: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['MotorFault_model_2_t'][constants.TARGET_DEVICE_F280013]),
                constants.TARGET_DEVICE_F280015: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['MotorFault_model_2_t'][constants.TARGET_DEVICE_F280015]),
                constants.TARGET_DEVICE_F28003: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['MotorFault_model_2_t'][constants.TARGET_DEVICE_F28003]),
                constants.TARGET_DEVICE_F28004: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['MotorFault_model_2_t'][constants.TARGET_DEVICE_F28004]),
                constants.TARGET_DEVICE_F2837: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['MotorFault_model_2_t'][constants.TARGET_DEVICE_F2837]),
                constants.TARGET_DEVICE_F28P65: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['MotorFault_model_2_t'][constants.TARGET_DEVICE_F28P65]),
                constants.TARGET_DEVICE_F28P55: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['MotorFault_model_2_t'][constants.TARGET_DEVICE_F28P55]),
                constants.TARGET_DEVICE_MSPM0G3507: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['MotorFault_model_2_t'][constants.TARGET_DEVICE_MSPM0G3507]),
                constants.TARGET_DEVICE_MSPM0G5187: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['MotorFault_model_2_t'][constants.TARGET_DEVICE_MSPM0G5187]),
            },
            properties=[dict(type="group", dynamic=True, script="motorfault.py", name="preprocessing_group", label="Preprocessing Parameters", default=[])] + template_gui_model_properties
        ),
    }),
    'MotorFault_model_3_t': utils.deep_update_dict(deepcopy(template_model_description), {
        'common': dict(
            task_type=constants.TASK_TYPE_MOTOR_FAULT,
            generic_model=False,
            model_details='TI\u2122 handcrafted model.\nClassification Model with 1k params.\nMiddle of the 3 CNN based models.',
            help_url="file://models/MotorFault_model_3_t/MotorFault_model_3_t.md"
        ),
        'training': dict(
            model_training_id='CNN_MF_3L',
            model_name='MotorFault_model_3_t',
            learning_rate=0.01,
            model_spec=os.path.join(repo_parent_path, 'tinyml-mlbackend', 'tinyml_proprietary_models', 'cnn_mf_3l.py'),
            batch_size=constants.TRAINING_BATCH_SIZE_DEFAULT[constants.TASK_TYPE_MOTOR_FAULT],
            target_devices={
                constants.TARGET_DEVICE_F280013: dict(model_selection_factor=1) | (DEVICE_RUN_INFO['MotorFault_model_3_t'][constants.TARGET_DEVICE_F280013]),
                constants.TARGET_DEVICE_F280015: dict(model_selection_factor=1) | (DEVICE_RUN_INFO['MotorFault_model_3_t'][constants.TARGET_DEVICE_F280015]),
                constants.TARGET_DEVICE_F28003: dict(model_selection_factor=1) | (DEVICE_RUN_INFO['MotorFault_model_3_t'][constants.TARGET_DEVICE_F28003]),
                constants.TARGET_DEVICE_F28004: dict(model_selection_factor=1) | (DEVICE_RUN_INFO['MotorFault_model_3_t'][constants.TARGET_DEVICE_F28004]),
                constants.TARGET_DEVICE_F2837: dict(model_selection_factor=1) | (DEVICE_RUN_INFO['MotorFault_model_3_t'][constants.TARGET_DEVICE_F2837]),
                constants.TARGET_DEVICE_F28P65: dict(model_selection_factor=1) | (DEVICE_RUN_INFO['MotorFault_model_3_t'][constants.TARGET_DEVICE_F28P65]),
                constants.TARGET_DEVICE_F28P55: dict(model_selection_factor=1) | (DEVICE_RUN_INFO['MotorFault_model_3_t'][constants.TARGET_DEVICE_F28P55]),
                constants.TARGET_DEVICE_MSPM0G3507: dict(model_selection_factor=1)  | (DEVICE_RUN_INFO['MotorFault_model_3_t'][constants.TARGET_DEVICE_MSPM0G3507]),
                constants.TARGET_DEVICE_MSPM0G5187: dict(model_selection_factor=1) | (DEVICE_RUN_INFO['MotorFault_model_3_t'][constants.TARGET_DEVICE_MSPM0G5187]),
            },
            properties=[dict(type="group", dynamic=True, script="motorfault.py", name="preprocessing_group", label="Preprocessing Parameters", default=[])] + template_gui_model_properties
        ),
    }),
    'FanImbalance_model_1_t': utils.deep_update_dict(deepcopy(template_model_description), {
        'common': dict(
            task_type=constants.TASK_TYPE_BLOWER_IMBALANCE,
            generic_model=False,
            model_details='TI\u2122 handcrafted model.\nClassification Model with ~0.6k params.\nSimplest model.',
            #help_url="file://models/MotorFault_model_1_t/MotorFault_model_1_t.md"
        ),
        'training': dict(
            model_training_id='CNN_MF_1L',
            model_name='FanImbalance_model_1_t',
            learning_rate=0.01,
            model_spec=os.path.join(repo_parent_path, 'tinyml-mlbackend', 'tinyml_proprietary_models', 'cnn_mf_1l.py'),
            batch_size=constants.TRAINING_BATCH_SIZE_DEFAULT[constants.TASK_TYPE_BLOWER_IMBALANCE],
            target_devices={
                constants.TARGET_DEVICE_F280013: dict(model_selection_factor=2) | (DEVICE_RUN_INFO['FanImbalance_model_1_t'][constants.TARGET_DEVICE_F280013]),
                constants.TARGET_DEVICE_F280015: dict(model_selection_factor=2) | (DEVICE_RUN_INFO['FanImbalance_model_1_t'][constants.TARGET_DEVICE_F280015]),
                constants.TARGET_DEVICE_F28003: dict(model_selection_factor=2) | (DEVICE_RUN_INFO['FanImbalance_model_1_t'][constants.TARGET_DEVICE_F28003]),
                constants.TARGET_DEVICE_F28004: dict(model_selection_factor=2) | (DEVICE_RUN_INFO['FanImbalance_model_1_t'][constants.TARGET_DEVICE_F28004]),
                constants.TARGET_DEVICE_F2837: dict(model_selection_factor=2) | (DEVICE_RUN_INFO['FanImbalance_model_1_t'][constants.TARGET_DEVICE_F2837]),
                constants.TARGET_DEVICE_F28P65: dict(model_selection_factor=2) | (DEVICE_RUN_INFO['FanImbalance_model_1_t'][constants.TARGET_DEVICE_F28P65]),
                constants.TARGET_DEVICE_F28P55: dict(model_selection_factor=2) | (DEVICE_RUN_INFO['FanImbalance_model_1_t'][constants.TARGET_DEVICE_F28P55]),
            },
            properties=[dict(type="group", dynamic=True, script="motorfault.py", name="preprocessing_group", label="Preprocessing Parameters", default=[])] + template_gui_model_properties
        ),
    }),
    'FanImbalance_model_2_t': utils.deep_update_dict(deepcopy(template_model_description), {
        'common': dict(
            task_type=constants.TASK_TYPE_BLOWER_IMBALANCE,
            generic_model=False,
            model_details='TI\u2122 handcrafted model.\nClassification Model with 3k params.\nBest & largest of the 3 models, hardest to train.',
            #help_url="file://models/MotorFault_model_2_t/MotorFault_model_2_t.md"
        ),
        'training': dict(
            model_training_id='CNN_MF_2L',
            model_name='FanImbalance_model_2_t',
            learning_rate=0.01,
            model_spec=os.path.join(repo_parent_path, 'tinyml-mlbackend', 'tinyml_proprietary_models', 'cnn_mf_2l.py'),
            batch_size=constants.TRAINING_BATCH_SIZE_DEFAULT[constants.TASK_TYPE_BLOWER_IMBALANCE],
            target_devices={
                constants.TARGET_DEVICE_F280013: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FanImbalance_model_2_t'][constants.TARGET_DEVICE_F280013]),
                constants.TARGET_DEVICE_F280015: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FanImbalance_model_2_t'][constants.TARGET_DEVICE_F280015]),
                constants.TARGET_DEVICE_F28003: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FanImbalance_model_2_t'][constants.TARGET_DEVICE_F28003]),
                constants.TARGET_DEVICE_F28004: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FanImbalance_model_2_t'][constants.TARGET_DEVICE_F28004]),
                constants.TARGET_DEVICE_F2837: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FanImbalance_model_2_t'][constants.TARGET_DEVICE_F2837]),
                constants.TARGET_DEVICE_F28P65: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FanImbalance_model_2_t'][constants.TARGET_DEVICE_F28P65]),
                constants.TARGET_DEVICE_F28P55: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['FanImbalance_model_2_t'][constants.TARGET_DEVICE_F28P55]),
            },
            properties=[dict(type="group", dynamic=True, script="motorfault.py", name="preprocessing_group", label="Preprocessing Parameters", default=[])] + template_gui_model_properties
        ),
    }),
    'FanImbalance_model_3_t': utils.deep_update_dict(deepcopy(template_model_description), {
        'common': dict(
            task_type=constants.TASK_TYPE_BLOWER_IMBALANCE,
            generic_model=False,
            model_details='TI\u2122 handcrafted model.\nClassification Model with 1k params.\nMiddle of the 3 CNN based models.',
            #help_url="file://models/MotorFault_model_3_t/MotorFault_model_3_t.md"
        ),
        'training': dict(
            model_training_id='CNN_MF_3L',
            model_name='FanImbalance_model_3_t',
            learning_rate=0.01,
            model_spec=os.path.join(repo_parent_path, 'tinyml-mlbackend', 'tinyml_proprietary_models', 'cnn_mf_3l.py'),
            batch_size=constants.TRAINING_BATCH_SIZE_DEFAULT[constants.TASK_TYPE_BLOWER_IMBALANCE],
            target_devices={
                constants.TARGET_DEVICE_F280013: dict(model_selection_factor=1) | (DEVICE_RUN_INFO['FanImbalance_model_3_t'][constants.TARGET_DEVICE_F280013]),
                constants.TARGET_DEVICE_F280015: dict(model_selection_factor=1) | (DEVICE_RUN_INFO['FanImbalance_model_3_t'][constants.TARGET_DEVICE_F280015]),
                constants.TARGET_DEVICE_F28003: dict(model_selection_factor=1) | (DEVICE_RUN_INFO['FanImbalance_model_3_t'][constants.TARGET_DEVICE_F28003]),
                constants.TARGET_DEVICE_F28004: dict(model_selection_factor=1) | (DEVICE_RUN_INFO['FanImbalance_model_3_t'][constants.TARGET_DEVICE_F28004]),
                constants.TARGET_DEVICE_F2837: dict(model_selection_factor=1) | (DEVICE_RUN_INFO['FanImbalance_model_3_t'][constants.TARGET_DEVICE_F2837]),
                constants.TARGET_DEVICE_F28P65: dict(model_selection_factor=1) | (DEVICE_RUN_INFO['FanImbalance_model_3_t'][constants.TARGET_DEVICE_F28P65]),
                constants.TARGET_DEVICE_F28P55: dict(model_selection_factor=1) | (DEVICE_RUN_INFO['FanImbalance_model_3_t'][constants.TARGET_DEVICE_F28P55]),
            },
            properties=[dict(type="group", dynamic=True, script="motorfault.py", name="preprocessing_group", label="Preprocessing Parameters", default=[])] + template_gui_model_properties
        ),
    }),
    #######################################
    'NAS': utils.deep_update_dict(deepcopy(template_model_description), {
        'common': dict(model_details='Model will be automatically found through NAS framework'),
        'training': dict(
            model_training_id='None',
            model_name='NAS',
            learning_rate=0.01,
            model_spec='',
            batch_size=constants.TRAINING_BATCH_SIZE_DEFAULT[constants.TASK_TYPE_MOTOR_FAULT],
            target_devices={
                constants.TARGET_DEVICE_F280013: dict(model_selection_factor=1) | (DEVICE_RUN_INFO['NAS'][constants.TARGET_DEVICE_F280013]),
                constants.TARGET_DEVICE_F280015: dict(model_selection_factor=1) | (DEVICE_RUN_INFO['NAS'][constants.TARGET_DEVICE_F280015]),
                constants.TARGET_DEVICE_F28003: dict(model_selection_factor=1) | (DEVICE_RUN_INFO['NAS'][constants.TARGET_DEVICE_F28003]),
                constants.TARGET_DEVICE_F28004: dict(model_selection_factor=1) | (DEVICE_RUN_INFO['NAS'][constants.TARGET_DEVICE_F28004]),
                constants.TARGET_DEVICE_F2837: dict(model_selection_factor=1) | (DEVICE_RUN_INFO['NAS'][constants.TARGET_DEVICE_F2837]),
                constants.TARGET_DEVICE_F28P65: dict(model_selection_factor=1) | (DEVICE_RUN_INFO['NAS'][constants.TARGET_DEVICE_F28P65]),
                constants.TARGET_DEVICE_F28P55: dict(model_selection_factor=1) | (DEVICE_RUN_INFO['NAS'][constants.TARGET_DEVICE_F28P55]),
                constants.TARGET_DEVICE_CC2755: dict(model_selection_factor=1) | (DEVICE_RUN_INFO['NAS'][constants.TARGET_DEVICE_CC2755]),
            },
        ),
    }),
    'PIRDetection_model_1_t': utils.deep_update_dict(deepcopy(template_model_description), {
        'common': dict(
            task_type=constants.TASK_TYPE_PIR_DETECTION,
            generic_model=False,
            model_details='TI\u2122 handcrafted model.\nClassification Model with 53k+ params.\n2-D CNN based model for multiple motion source detection.',
            help_url="file://models/PIRDetection_model_1_t/PIRDetection_model_1_t.md"
        ),
        'training': dict(
            model_training_id='CNN_TS_PIR2D_BASE',
            model_name='PIRDetection_model_1_t',
            learning_rate=0.04,
            batch_size=constants.TRAINING_BATCH_SIZE_DEFAULT[constants.TASK_TYPE_PIR_DETECTION],
            target_devices={
                #constants.TARGET_DEVICE_F280013: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['PIRDetection_model_1_t'][constants.TARGET_DEVICE_F280013]),
                #constants.TARGET_DEVICE_F280015: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['PIRDetection_model_1_t'][constants.TARGET_DEVICE_F280015]),
                #constants.TARGET_DEVICE_F28003: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['PIRDetection_model_1_t'][constants.TARGET_DEVICE_F28003]),
                #constants.TARGET_DEVICE_F28004: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['PIRDetection_model_1_t'][constants.TARGET_DEVICE_F28004]),
                #constants.TARGET_DEVICE_F2837: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['PIRDetection_model_1_t'][constants.TARGET_DEVICE_F2837]),
                #constants.TARGET_DEVICE_F28P65: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['PIRDetection_model_1_t'][constants.TARGET_DEVICE_F28P65]),
                #constants.TARGET_DEVICE_F28P55: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['PIRDetection_model_1_t'][constants.TARGET_DEVICE_F28P55]),
                constants.TARGET_DEVICE_CC2755: dict(model_selection_factor=0) | (DEVICE_RUN_INFO['PIRDetection_model_1_t'][constants.TARGET_DEVICE_CC2755]),
            },
            properties=[dict(type="group", dynamic=True, script="pirdetection.py", name="preprocessing_group", label="Preprocessing Parameters", default=[])] +
                       [template_gui_model_properties[0]] + [dict(label="Epochs", name="training_epochs", type="integer", default=50, min=2, max=1000),] + [template_gui_model_properties[2]]
        ),
    }),
}

enabled_models_list = [
    # 'TimeSeries_Generic_1k', 'TimeSeries_Generic_4k', 'TimeSeries_Generic_6k', 'TimeSeries_Generic_13k',
    'TimeSeries_Generic_100_t', 'TimeSeries_Generic_1k_t', 'TimeSeries_Generic_4k_t', 'TimeSeries_Generic_6k_t', 'TimeSeries_Generic_13k_t',
    'Res_Add_TimeSeries_Generic_3k_t', 'Res_Cat_TimeSeries_Generic_3k_t',
    'ArcFault_model_200_t', 'ArcFault_model_300_t', 'ArcFault_model_700_t', 'ArcFault_model_1400_t',
    'MotorFault_model_1_t', 'MotorFault_model_2_t', 'MotorFault_model_3_t', 'PIRDetection_model_1_t',
    'FanImbalance_model_1_t', 'FanImbalance_model_2_t', 'FanImbalance_model_3_t'
]


def get_model_descriptions(task_type=None):
    return _base_get_model_descriptions(_model_descriptions, enabled_models_list, task_type)


def get_model_description(model_name):
    return _base_get_model_description(_model_descriptions, enabled_models_list, model_name)


class ModelTraining(BaseModelTraining):
    _train_module = train
    _test_module = test

    def _get_extra_init_params(self):
        return dict(
            file_level_classification_log_path=os.path.join(
                self.params.training.train_output_path if self.params.training.train_output_path else self.params.training.training_path,
                'file_level_classification_summary.log'),
        )

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
                '--gof-test', f'{self.params.data_processing_feature_extraction.gof_test}',
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

                #######################################
                # nas params
                #######################################
                '--nas_enabled', f'{self.params.training.nas_enabled}',
                '--nas_optimization_mode', f'{self.params.training.nas_optimization_mode}',
                '--nas_model_size', f'{self.params.training.nas_model_size}',
                '--nas_epochs', f'{self.params.training.nas_epochs}',

                '--nas_nodes_per_layer', f'{self.params.training.nas_nodes_per_layer}',
                '--nas_layers', f'{self.params.training.nas_layers}',
                '--nas_init_channels', f'{self.params.training.nas_init_channels}',
                '--nas_init_channel_multiplier', f'{self.params.training.nas_init_channel_multiplier}',
                '--nas_fanout_concat', f'{self.params.training.nas_fanout_concat}',
                '--load_saved_model', f'{self.params.training.load_saved_model}',
                #######################################
                # end of nas params
                #######################################

                '--sampling-rate', f'{self.params.data_processing_feature_extraction.sampling_rate}',
                '--resampling-factor', f'{self.params.data_processing_feature_extraction.resampling_factor}',
                '--gain-variations', f'{self.params.data_processing_feature_extraction.gain_variations}',
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
                '--q15-scale-factor', f'{self.params.data_processing_feature_extraction.q15_scale_factor}',
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

                '--file-level-classification-log', f'{self.params.training.file_level_classification_log_path}',

                #'--tensorboard-logger', 'True',
                '--variables', f'{self.params.data_processing_feature_extraction.variables}',
                '--with-input-batchnorm', f'{self.params.training.with_input_batchnorm}',
                '--lis', f'{self.params.training.log_file_path}',
                # Do not add newer arguments after this line, it will change the behaviour of the code.
                '--data-path', os.path.join(self.params.dataset.dataset_path, self.params.dataset.data_dir),
                '--store-feat-ext-data', f'{self.params.data_processing_feature_extraction.store_feat_ext_data}',
                '--epochs', f'{self.params.training.training_epochs}',
                '--lr', f'{self.params.training.learning_rate}',
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
                '--gain-variations', f'{self.params.data_processing_feature_extraction.gain_variations}',
                '--stride-size', f'{self.params.data_processing_feature_extraction.stride_size}',

                '--data-proc-transforms', self.params.data_processing_feature_extraction.data_proc_transforms,
                '--feat-ext-transform', self.params.data_processing_feature_extraction.feat_ext_transform,
                # Arc Fault and Motor Fault Related Params
                '--frame-size', f'{self.params.data_processing_feature_extraction.frame_size}',
                '--feature-size-per-frame', f'{self.params.data_processing_feature_extraction.feature_size_per_frame}',
                '--num-frame-concat', f'{self.params.data_processing_feature_extraction.num_frame_concat}',
                '--q15-scale-factor', f'{self.params.data_processing_feature_extraction.q15_scale_factor}',
                '--frame-skip', f'{self.params.data_processing_feature_extraction.frame_skip}',
                '--min-bin', f'{self.params.data_processing_feature_extraction.min_bin}',
                '--normalize-bin', f'{self.params.data_processing_feature_extraction.normalize_bin}',
                '--dc-remove', f'{self.params.data_processing_feature_extraction.dc_remove}',
                '--analysis-bandwidth', f'{self.params.data_processing_feature_extraction.analysis_bandwidth}',
                # PIR Detection related params
                '--window-count', f'{self.params.data_processing_feature_extraction.window_count}',
                '--chunk-size', f'{self.params.data_processing_feature_extraction.chunk_size}',
                '--fft-size',f'{self.params.data_processing_feature_extraction.fft_size}',
                # End of PIR Detection related params
                # '--num-channel', f'{self.params.data_processing_feature_extraction.num_channel}',
                '--log-base', f'{self.params.data_processing_feature_extraction.log_base}',
                '--log-mul', f'{self.params.data_processing_feature_extraction.log_mul}',
                '--log-threshold', f'{self.params.data_processing_feature_extraction.log_threshold}',
                '--stacking', f'{self.params.data_processing_feature_extraction.stacking}',
                '--offset', f'{self.params.data_processing_feature_extraction.offset}',
                '--scale', f'{self.params.data_processing_feature_extraction.scale}',
                '--nn-for-feature-extraction', f'{self.params.data_processing_feature_extraction.nn_for_feature_extraction}',
                '--output-dequantize', f'{self.params.training.output_dequantize}',

                '--file-level-classification-log', f'{self.params.training.file_level_classification_log_path}',

                # '--tensorboard-logger', 'True',
                '--lis', f'{self.params.training.log_file_path}',
                '--data-path', f'{data_path}',
                '--output-dir', output_dir,
                '--model-path', f'{model_path}',
                '--generic-model', f'{self.params.common.generic_model}',
                ]
