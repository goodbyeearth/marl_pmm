REGISTRY = {}

from .basic_controller import BasicMAC
from .test_controller import TestMAC
from .see_id_controller import SeeIdMAC
from .test_controller_see_id import TestSeeIdMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY['test_mac'] = TestMAC
REGISTRY['see_id_mac'] = SeeIdMAC
REGISTRY['test_see_id_mac'] = TestSeeIdMAC