from __future__ import annotations
from abc import ABC, abstractmethod
import os
import torch
from torchtyping import TensorType
from src.gfn.containers.states import States
from src.gfn.containers.trajectories import Trajectories
from src.gfn.envs import Env

TensorFloat = TensorType["n_trajectories", torch.float]
class ReplayBuffer(ABC):
    def __init__(
        self,
        env: Env,
        capacity: int = 1000,
    ):
        self.env = env
        self.capacity = capacity
        self._is_full = False
        self._index = 0
    def __repr__(self):
        return f"ReplayBuffer(capacity={self.capacity}, containing {len(self)} training samples)"
    def __len__(self):
        return self.capacity if self._is_full else self._index

class Replay_x(ReplayBuffer):  # replay of index of terminating states
    def __init__(self,
                 env: Env,
                 capacity: int = int(1e10), ):
        super().__init__(env,capacity)
        self.x_index = torch.LongTensor(0)
        self.x_rewards = torch.FloatTensor(0)

    def __repr__(self):
        return f"ReplayBuffer(capacity={self.capacity}, containing {len(self)} terminating states)"

    def add(self, x_states: States, rewards: TensorFloat):
        to_add = len(x_states)
        self._is_full |= self._index + to_add >= self.capacity
        self._index = (self._index + to_add) % self.capacity
        #
        self.x_index = torch.cat((self.x_index, torch.tensor(self.env.get_terminating_states_indices(x_states))))
        self.x_rewards = torch.cat((self.x_rewards, rewards))
        self.x_index = self.x_index[-self.capacity:]
        self.x_rewards = self.x_rewards[-self.capacity:]

class  Replay_Traj(ReplayBuffer):
    def __init__(
        self,
        env: Env,
        capacity: int = 1000,
    ):
        super().__init__(env,capacity)
        self.training_objects = Trajectories(env)
    def __repr__(self):
        return f"ReplayBuffer(capacity={self.capacity}, containing {len(self)} trajectories)"
    def __len__(self):
        return self.capacity if self._is_full else self._index

    def add(self, training_objects: Trajectories):
        to_add = len(training_objects)
        self._is_full |= self._index + to_add >= self.capacity
        self._index = (self._index + to_add) % self.capacity

        self.training_objects.extend(training_objects)
        self.training_objects = self.training_objects[-self.capacity :]

    def sample(self, n_trajectories: int) -> Trajectories:
        samples=self.training_objects.sample(n_trajectories)
        return samples

    def save(self, directory: str):
        self.training_objects.save(os.path.join(directory, "training_objects"))

    def load(self, directory: str):
        self.training_objects.load(os.path.join(directory, "training_objects"))
        self._index = len(self.training_objects)
