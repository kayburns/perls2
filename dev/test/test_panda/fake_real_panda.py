"""Class definition for Fake RealPandaInterface
"""

import pytest
from perls2.utils.yaml_config import YamlConfig
from perls2.redis_interfaces.redis_keys import *
from perls2.robots.real_panda_interface import RealPandaInterface

class FakePandaInterface(RealPandaInterface):
    config = YamlConfig('dev/test/test_panda/test_panda_cfg.yaml')

    def __init__(self,
        config= YamlConfig('dev/test/test_panda/test_panda_cfg.yaml'),
        controlType="EEImpedance"):
        super().__init__(config, controlType)
