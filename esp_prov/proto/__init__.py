# SPDX-FileCopyrightText: 2018-2022 Espressif Systems (Shanghai) CO LTD
# SPDX-License-Identifier: Apache-2.0
#

import importlib.util
import os
import sys
from importlib.abc import Loader
from typing import Any


def _load_source(name: str, path: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if not spec:
        return None

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert isinstance(spec.loader, Loader)
    spec.loader.exec_module(module)
    return module


# Get the current directory path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
parent_dir_1 = os.path.dirname(parent_dir)
print(parent_dir_1)

# protocomm component related python files generated from .proto files
constants_pb2 = _load_source('constants_pb2', parent_dir_1 + '/protocomm/constants_pb2.py')
sec1_pb2      = _load_source('sec1_pb2',      parent_dir_1 + '/protocomm/sec1_pb2.py')
sec0_pb2      = _load_source('sec0_pb2',      parent_dir_1 + '/protocomm/sec0_pb2.py')
sec2_pb2      = _load_source('sec2_pb2',      parent_dir_1 + '/protocomm/sec2_pb2.py')
session_pb2   = _load_source('session_pb2',   parent_dir_1 + '/protocomm/session_pb2.py')

# wifi_provisioning component related python files generated from .proto files
wifi_constants_pb2 = _load_source('wifi_constants_pb2', parent_dir_1 + '/provisioning/wifi_constants_pb2.py')
wifi_config_pb2    = _load_source('wifi_config_pb2',    parent_dir_1 + '/provisioning/wifi_config_pb2.py')
wifi_scan_pb2      = _load_source('wifi_scan_pb2',      parent_dir_1 + '/provisioning/wifi_scan_pb2.py')
wifi_ctrl_pb2    = _load_source('wifi_ctrl_pb2',    parent_dir_1 + '/provisioning/wifi_ctrl_pb2.py')
