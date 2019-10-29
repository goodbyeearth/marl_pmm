import pommerman
from pommerman import agents


def get_env_fn():
    def _thunk():
        agent_list = [
            agents.SimpleAgent(),
            # agents.RandomAgent(),
            agents.SimpleAgent(),
            agents.SimpleAgent(),
            agents.SimpleAgent()
            # agents.RandomAgent(),
        ]
        env = pommerman.make("PommeRadioCompetition-v2", agent_list)
        return env
    return _thunk


REGISTRY = {}
REGISTRY["PommeRadioCompetition-v2"] = get_env_fn()


