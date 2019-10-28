REGISTRY = {}

from .rnn_agent import RNNAgent
from .pmm_agent import PmmAgent

REGISTRY["rnn"] = RNNAgent
REGISTRY["pmm"] = PmmAgent