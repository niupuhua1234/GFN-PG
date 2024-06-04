from __future__ import annotations

import os

from src.gfn.containers.states import States
from src.gfn.containers.trajectories import Trajectories
from src.gfn.containers.transitions import Transitions
from src.gfn.losses.base import (
    EdgeDecomposableLoss,
    Loss,
    StateDecomposableLoss,
    TrajectoryDecomposableLoss,
)
from src.gfn.envs import Env

class ReplayBuffer:
    def __init__(
        self,
        env: Env,
        loss_fn: Loss | None = None,
        capacity: int = 1000,
    ):
        self.env = env
        self.capacity = capacity
        self.loss_fn=loss_fn
        self._is_full = False
        self._index = 0
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

    def sample(self, n_trajectories: int) -> Transitions | Trajectories | States:

        samples=self.training_objects.sample(n_trajectories)
        if isinstance(self.loss_fn, StateDecomposableLoss):
            return samples.intermediary_states
        elif isinstance(self.loss_fn, EdgeDecomposableLoss):
            return samples.to_transitions()
        elif isinstance(self.loss_fn, TrajectoryDecomposableLoss):
            return samples
        else:
            raise ValueError(f"Loss {self.loss_fn} is not supported.")

    def save(self, directory: str):
        self.training_objects.save(os.path.join(directory, "training_objects"))

    def load(self, directory: str):
        self.training_objects.load(os.path.join(directory, "training_objects"))
        self._index = len(self.training_objects)
