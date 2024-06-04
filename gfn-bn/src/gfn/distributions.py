from abc import ABC, abstractmethod
from collections import Counter
from typing import Optional

import torch
from torchtyping import TensorType

from src.gfn.containers.states import States
from src.gfn.envs.env import Env
# Typing
TensorPmf = TensorType["n_states", float]



class TerminatingStatesDist(ABC):
    """
    Represents an abstract distribution over terminating states.
    """

    @abstractmethod
    def pmf(self,states:States) -> TensorPmf:
        """
        Compute the probability mass function of the distribution.
        """
        pass


class Empirical_Dist(TerminatingStatesDist):
    """
    Represents an empirical distribution over terminating states.
    """
    def __init__(self, env: Env) -> None:
        self.states_to_indices    = env.get_terminating_states_indices
        self.env_n_terminating_states = env.n_terminating_states

    def pmf(self,states) -> TensorPmf:
        assert len(states.batch_shape) == 1, "States should be a linear batch of states"
        states_indices = self.states_to_indices(states)
        counter = Counter(states_indices)
        counter_list = [counter[state_idx] if state_idx in counter else 0
                        for state_idx in range(self.env_n_terminating_states)]
        return torch.tensor(counter_list, dtype=torch.float) / len(states_indices)

class Empirical_Ratio(TerminatingStatesDist):
    """
    Represents an empirical distribution over terminating states.
    """
    def __init__(self, env: Env) -> None:
        self.states_to_indices=env.empirical_states_indices
        self.env_n_terminating_states = env.empirical_n_terminating_states

    def pmf(self,states) -> TensorPmf:
        assert len(states.batch_shape) == 1, "States should be a linear batch of states"
        states_indices = self.states_to_indices(states)
        counter = Counter(states_indices)
        counter_list = [counter[state_idx] if state_idx in counter else 0
                        for state_idx in range(self.env_n_terminating_states(states))]
        return torch.tensor(counter_list, dtype=torch.float) / len(states_indices)

