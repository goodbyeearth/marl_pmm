REGISTRY = {}

from .basic_controller import BasicMAC
from .test_controller import TestMAC
from .two_agent_controller import TwoAgentMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY['test_mac'] = TestMAC
REGISTRY['two_agent_mac'] = TwoAgentMAC