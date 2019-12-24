#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import op_test
import unittest
import numpy
import paddle.fluid as fluid
from paddle.fluid import Program, program_guard


def create_test_class(op_type, typename, callback, force_cpu=None):
    class Cls(op_test.OpTest):
        def setUp(self):
            a = numpy.random.random(size=(10, 7)).astype(typename)
            b = numpy.random.random(size=(10, 7)).astype(typename)
            c = callback(a, b)
            self.inputs = {'X': a, 'Y': b}
            self.outputs = {'Out': c}
            self.op_type = op_type
            self.attrs = {'force_cpu': force_cpu}

        def test_output(self):
            self.check_output()

    cls_name = "{0}_{1}_{2}".format(op_type, typename, force_cpu)
    Cls.__name__ = cls_name
    globals()[cls_name] = Cls

def create_cuda_test_class(op_type, typename, callback, force_cpu=None):
    class Cls(op_test.OpTest):
        def setUp(self):
            a = numpy.random.random(size=(10, 7)).astype(typename)
            b = numpy.random.random(size=(10, 7)).astype(typename)
            c = callback(a, b)
            self.inputs = {'X': a, 'Y': b}
            self.outputs = {'Out': c}
            self.op_type = op_type
            self.attrs = {'force_cpu': force_cpu}

        def test_output(self):
            if core.is_compiled_with_cuda():
                place = core.CUDAPlace(0)
                self.check_output_with_place(place, atol=1e-5)

    cls_name = "{0}_{1}_{2}".format(op_type, typename, force_cpu)
    Cls.__name__ = cls_name
    globals()[cls_name] = Cls

for _type_name in {'float32', 'float64', 'int32', 'int64'}:
    create_test_class('less_than', _type_name, lambda _a, _b: _a < _b)
    create_test_class('less_equal', _type_name, lambda _a, _b: _a <= _b)
    create_test_class('greater_than', _type_name, lambda _a, _b: _a > _b)
    create_test_class('greater_equal', _type_name, lambda _a, _b: _a >= _b)
    create_test_class('equal', _type_name, lambda _a, _b: _a == _b)
    create_test_class('not_equal', _type_name, lambda _a, _b: _a != _b)

for _type_name in {'float32', 'float64', 'int32', 'int64'}:
    create_test_class('less_than', _type_name, lambda _a, _b: _a < _b, False)
    create_test_class('less_equal', _type_name, lambda _a, _b: _a <= _b, False)
    create_test_class('greater_than', _type_name, lambda _a, _b: _a > _b, False)
    create_test_class('greater_equal', _type_name, lambda _a, _b: _a >= _b, False)
    create_test_class('equal', _type_name, lambda _a, _b: _a == _b, False)
    create_test_class('not_equal', _type_name, lambda _a, _b: _a != _b, False)

for _type_name in {'float32', 'float64', 'int32', 'int64'}:
    create_cuda_test_class('less_than', _type_name, lambda _a, _b: _a < _b, False)
    create_cuda_test_class('less_equal', _type_name, lambda _a, _b: _a <= _b, False)
    create_cuda_test_class('greater_than', _type_name, lambda _a, _b: _a > _b, False)
    create_cuda_test_class('greater_equal', _type_name, lambda _a, _b: _a >= _b, False)
    create_cuda_test_class('equal', _type_name, lambda _a, _b: _a == _b, False)
    create_cuda_test_class('not_equal', _type_name, lambda _a, _b: _a != _b, False)

class TestCompareOpError(unittest.TestCase):
    def test_errors(self):
        with program_guard(Program(), Program()):
            # The input x and y of compare_op must be Variable.
            x = fluid.layers.data(name='x', shape=[1], dtype="float32")
            y = fluid.create_lod_tensor(
                numpy.array([[-1]]), [[1]], fluid.CPUPlace())
            self.assertRaises(TypeError, fluid.layers.greater_equal, x, y)


if __name__ == '__main__':
    unittest.main()
