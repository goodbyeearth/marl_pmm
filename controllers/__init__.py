REGISTRY = {}

from .basic_controller import BasicMAC
from .test_controller import TestMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY['test_mac'] = TestMAC