#
# Copyright (c) 2019-2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from pyspark.sql.types import *

label = 'label'
features = [ 'feature_' + str(i) for i in range(0, 126) ]
schema = StructType([ StructField(x, FloatType()) for x in [label] + features ])

default_params = {
    'eta': 0.1,
    'missing': 0.0,
    'maxDepth': 2,
    'numWorkers': 1,
}
