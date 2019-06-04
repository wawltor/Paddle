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

import unittest
import numpy as np
from op_test import OpTest


class TestMatMulSparse(OpTest):
    def setUp(self):
        self.op_type = "matmul_sparse"
        self.config()
        tmp_data = np.random.randint(1,5)
        example = np.arange(24).reshape(6, 4).astype(np.float32)
        row_ids = np.array([0, 1, 2])
        col_ids = np.array([0, 1, 2, 3, 0])
        sp_row_num = np.array([3])
        lod = [[2, 2, 1]]
        self.inputs = {
            'X': example,
            'RowIds': row_ids,
            'ColIds': (col_ids, lod)
            'Sp_row_num': sp_row_num
        }
        output = np.array([4, 6, 8, 10, 20, 22, 24, 26, 0, 1, 2, 3]).reshape(3, 4).astype(np.float32) 
        self.outputs = {'Out': output}

    #def test_check_output(self):
    #    self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out',  no_grad_set=set(['RowIds', 'ColIds'])) 

if __name__ == "__main__":
    unittest.main()
