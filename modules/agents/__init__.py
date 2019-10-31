REGISTRY = {}

from .rnn_agent import RNNAgent
from .pmm_agent import PmmAgent
from .agent_see_id import SeeIdAgent

REGISTRY["rnn"] = RNNAgent
REGISTRY["pmm"] = PmmAgent
REGISTRY['see_id_agent'] = SeeIdAgent