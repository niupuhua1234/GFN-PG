from typing import Tuple

import torch
from torchtyping import TensorType
from src.gfn.containers import Trajectories
from src.gfn.estimators import LogZEstimator
from src.gfn.losses.base import PFBasedParametrization, TrajectoryDecomposableLoss


# Typing
ScoresTensor = TensorType["n_trajectories", float]
LossTensor = TensorType[0, float]
LogPTrajectoriesTensor = TensorType["max_length", "n_trajectories", float]

class TBParametrization(PFBasedParametrization):
    r"""
    :math:`\mathcal{O}_{PFZ} = \mathcal{O}_1 \times \mathcal{O}_2 \times \mathcal{O}_3`, where
    :math:`\mathcal{O}_1 = \mathbb{R}` represents the possible values for logZ,
    and :math:`\mathcal{O}_2` is the set of forward probability functions consistent with the DAG.
    :math:`\mathcal{O}_3` is the set of backward probability functions consistent with the DAG, or a singleton
    thereof, if self.logit_PB is a fixed LogitPBEstimator.
    Useful for the Trajectory Balance Loss.
    """
    def __init__(self, logit_PF,logit_PB, logZ: LogZEstimator):
        self.logZ=logZ
        super().__init__(logit_PF,logit_PB)

class TrajectoryBalance(TrajectoryDecomposableLoss):
    def __init__(
        self,
        parametrization: TBParametrization,
        optimizer:torch.optim.Optimizer,
        log_reward_clip_min: float = -12,
    ):
        """Loss object to evaluate the TB loss on a batch of trajectories.

        Args:
            log_reward_clip_min (float, optional): minimal value to clamp the reward to. Defaults to -12 (roughly log(1e-5)).
            on_policy (bool, optional): If True, the log probs stored in the trajectories are used. Defaults to False.

        Forward     P(·|s0)→  ...  →P(sT|sn-1) → P(sf|sT)
                                                     =1  for graded DAG
        -------------------------------------------------
        Backward    P(s0|s1)←  ... ←P(sn-1|sT) ← P(sT|sf)
                      =1                           R(sT)/Z
        """
        super().__init__(parametrization)
        self.log_reward_clip_min = log_reward_clip_min
        self.optimizer=optimizer

    def get_scores(self, trajectories: Trajectories) -> Tuple[LogPTrajectoriesTensor,LogPTrajectoriesTensor,LogPTrajectoriesTensor]:

        log_pf_trajectories = self.get_pfs(trajectories)
        log_pb_trajectories = self.get_pbs(trajectories)
        terminal_index=trajectories.is_terminating_action
        #
        pred  =log_pf_trajectories
        pred[0,:] +=self.parametrization.logZ.tensor
        target=log_pb_trajectories
        target.T[terminal_index.T]+=trajectories.log_rewards.clamp_min(self.log_reward_clip_min)
        return (log_pf_trajectories,
                log_pb_trajectories,
                pred-target,)

    def update_model(self,trajectories: Trajectories):
        loss=self.__call__(trajectories)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def __call__(self, trajectories: Trajectories) -> LossTensor:
        _, _,score= self.get_scores(trajectories)
        loss=score.sum(0).pow(2).mean()
        if torch.isnan(loss).any():raise ValueError("loss is nan")
        return loss

