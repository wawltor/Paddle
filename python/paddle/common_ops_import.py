# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from six.moves import reduce
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.framework import convert_np_dtype_to_dtype_, in_dygraph_mode, _varbase_creator
from paddle.fluid.framework import Variable, device_guard
from paddle.fluid.initializer import Constant
from paddle.fluid.core import VarDesc
from paddle.fluid import core
from paddle.fluid.data_feeder import check_type, check_dtype, convert_dtype, check_variable_and_dtype
from paddle.fluid.layers import utils
import paddle.fluid as fluid
import numpy
import warnings
