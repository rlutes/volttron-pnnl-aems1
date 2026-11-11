# -*- coding: utf-8 -*- {{{
# ===----------------------------------------------------------------------===
#
#                 AEMS Application
#
# ===----------------------------------------------------------------------===
#
# Copyright 2024 Battelle Memorial Institute
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy
# of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.
#
# ===----------------------------------------------------------------------===
# }}}
import json
import sys
from pathlib import Path
from unittest import mock

import pytest

Manager_dir = Path(__file__).parent.parent / 'Manager'
# Example agent configuration shipped with the repo (Schneider thermostat).
EXAMPLE_CONFIG = Path(__file__).parent.parent / 'configurations' / 'thermostats' / 'schneider.config'

# Command-line tools for a live platform (need the AEMS client library and run
# on their own, see README.md) - not pytest tests, so don't collect them.
collect_ignore = ['test_platform_driver.py', 'test_rpc_agent.py']

# Includes Manager package in the sys.path so we can do import manager and
# it will resolve to the manager package in the Manager folder
sys.path.insert(0, Manager_dir.as_posix())

from manager.main_manager import ManagerProxy


class _FakeConfigStore:
    """Empty VOLTTRON config store: every entry is missing until set."""

    def __init__(self):
        self.entries = {}

    def get(self, name):
        return self.entries[name]                 # KeyError when missing, like vip.config.get

    def set(self, name, contents, **kwargs):
        self.entries[name] = contents


class FakeAgent:
    """Stands in for ManagerAgent so ManagerProxy can be built without a platform."""

    def __init__(self, proxy, default_config, **kwargs):
        self.core = mock.MagicMock(identity="manager.test")
        self.core.schedule.return_value = mock.MagicMock()
        self.vip = mock.MagicMock()
        self.vip.config = _FakeConfigStore()


@pytest.fixture
def manager_proxy(tmp_path) -> ManagerProxy:
    # Copy the shipped example config, keeping data and models out of ~/.manager.
    config = json.loads(EXAMPLE_CONFIG.read_text())
    config.update(data_dir=str(tmp_path / "data"), model_dir=str(tmp_path / "models"))
    path = tmp_path / "manager.config"
    path.write_text(json.dumps(config))
    return ManagerProxy(config_path=path.as_posix(), agent_to_proxy=FakeAgent)
