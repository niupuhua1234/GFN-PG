from typing import List, Literal, Tuple

import torch
from torchtyping import TensorType

from src.gfn.containers import Trajectories
from src.gfn.estimators import LogStateFlowEstimator
from src.gfn.losses.base import PFBasedParametrization, TrajectoryDecomposableLoss

# Typing
ScoresTensor = TensorType[-1, float]
LossTensor = TensorType[0, float]
LogPTrajectoriesTensor = TensorType["max_length", "n_trajectories", float]

class SubTBParametrization(PFBasedParametrization):
    r"""
    Exactly the same as DBParametrization
    """
    def __init__(self, logit_PF,logit_PB, logF: LogStateFlowEstimator):
        self.logF=logF
        super().__init__(logit_PF,logit_PB)

class SubTrajectoryBalance(TrajectoryDecomposableLoss):
    def __init__(
        self,
        parametrization: SubTBParametrization,
        optimizer: torch.optim.Optimizer,
        log_reward_clip_min: float = -12,
        weighing: Literal["DB","ModifiedDB","TB",
                          "geometric","equal","geometric_within","equal_within"] = "equal_within",
        lamda: float = 0.9,
    ):
        """
        Args:
            parametrization: parametrization of the model
            log_reward_clip_min: minimum value of the log-reward. Log-Rewards lower than this value will be clipped to this value. Defaults to -12 (roughly log(1e-5)).
            weighing: how to weigh the different sub-trajectories of each trajectory.
                    - "DB": Considers all one-step transitions of each trajectory in the batch and weighs them equally (regardless of the length of trajectory).
                    Should be equivalent to DetailedBalance loss.
                    - "ModifiedDB": Considers all one-step transitions of each trajectory in the batch and weighs them inversely proportional to the trajectory length.
                            This ensures that the loss is not dominated by long trajectories. Each trajectory contributes equally to the loss.
                    - "TB": Considers only the full trajectory. Should be equivalent to TrajectoryBalance loss.
                    - "equal_within": Each sub-trajectory of each trajectory is weighed equally within the trajectory. Then each trajectory is weighed equally within the batch.
                    - "equal": Each sub-trajectory of each trajectory is weighed equally within the set of all sub-trajectories.
                    - "geometric_within": Each sub-trajectory of each trajectory is weighed proportionally to (lamda ** len(sub_trajectory)), within each trajectory.
                    - "geometric": Each sub-trajectory of each trajectory is weighed proportionally to (lamda ** len(sub_trajectory)), within the set of all sub-trajectories.
            lamda: parameter for geometric weighing
        """
        # Lamda is a discount factor for longer trajectories. The part of the loss
        # corresponding to sub-trajectories of length i is multiplied by lamda^i
        # where an edge is of length 1. As lamda approaches 1, each loss becomes equally weighted.
        super().__init__(parametrization)
        self.weighing = weighing
        self.lamda = lamda
        self.log_reward_clip_min = log_reward_clip_min
        self.optimizer=optimizer

    def get_scores(
        self, trajectories: Trajectories
    ) -> Tuple[List[ScoresTensor], List[ScoresTensor]]:
        """
        Returns two elements:
        - A list of tensors, each of which representing the scores of all sub-trajectories of length k, for k in [1, ..., trajectories.max_length].
            where the score of a sub-trajectory tau is log P_F(tau) + log F(tau_0) - log P_B(tau) - log F(tau_{-1}). The shape of the k-th tensor
            is (trajectories.max_length - k + 1, trajectories.n_trajectories), k starting from 1.
        - A list of tensors representing what should be masked out in each element of the first list, given that not all sub-trajectories
            of length k exist for each trajectory. The entries of those tensors are True if the corresponding sub-trajectory does not exist.
        """
        log_pf_trajectories = self.get_pfs(trajectories)
        log_pb_trajectories = self.get_pbs(trajectories)

        log_pf_trajectories_cum = self.cumulative_logprobs(trajectories, log_pf_trajectories)
        log_pb_trajectories_cum = self.cumulative_logprobs(trajectories, log_pb_trajectories)

        log_state_flows = torch.full_like(log_pf_trajectories, fill_value=self.fill_value)
        valid_mask = ~trajectories.is_sink_action
        valid_states = trajectories.states[:-1][valid_mask]
        log_state_flows[valid_mask] = self.parametrization.logF(valid_states).squeeze(-1)

        terminal_mask =  trajectories.is_terminating_action
        inter_mask = valid_mask & ~terminal_mask

        flattening_masks = []
        scores = []
        for j in range(trajectories.max_length): # j=0->(max_length-1)
            current_log_state_flows = (log_state_flows if j == 0 else log_state_flows[: -j])
            preds = (log_pf_trajectories_cum[j+1:]-log_pf_trajectories_cum[:-j-1]+ current_log_state_flows)
            #cum_logp:    i→.....sk...    →n
            #----------------------------------
            #cum_logp:    0→.....sk-i...  →n-i
            # logP   :τ(s0→si|s) Pτ(sn-i→sn|sn-i)
            targets = torch.full_like(preds, fill_value=self.fill_value)  #log P_B(sT->sf) + log F(sf) =logR(sT)
            targets.T[terminal_mask[j :].T] = trajectories.log_rewards[trajectories.when_is_done > j].clamp_min(self.log_reward_clip_min)
            # For now, the targets contain the log-rewards of the ending sub trajectories
            # We need to add to that the log-probabilities of the backward actions up-to the sub-trajectory's terminating state
            if j > 0:
                targets[terminal_mask[j :]] += (log_pb_trajectories_cum[j :] -log_pb_trajectories_cum[: -j])[:-1][terminal_mask[j :]] # note j here not j+1, since end=sT,  len(τ)-1 here
            # The following creates the targets for the non-finishing sub-trajectories
            targets[inter_mask[j :]] = (log_pb_trajectories_cum[j+1:] -log_pb_trajectories_cum[:-j-1])[inter_mask[j:]] + \
                                           log_state_flows[j+1:][valid_mask[j+1:]]

            flattening_mask = trajectories.when_is_done.lt(torch.arange(j+1,trajectories.max_length + 1,device=trajectories.when_is_done.device,).unsqueeze(-1))
            flat_preds = preds[~flattening_mask]   # masking  log_p cum for traj_length <j+1 keep  >=j+1
            flat_targets = targets[~flattening_mask]

            if torch.any(torch.isnan(flat_preds)): raise ValueError("NaN in preds")
            if torch.any(torch.isnan(flat_targets)):raise ValueError("NaN in targets")

            flattening_masks.append(flattening_mask)
            scores.append(preds - targets)

        return scores,flattening_masks

    def update_model(self,trajectories: Trajectories):
        loss=self.__call__(trajectories)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def __call__(self, trajectories: Trajectories) -> LossTensor:
        scores,flattening_masks= self.get_scores(trajectories)
        flattening_mask = torch.cat(flattening_masks)
        all_scores = torch.cat(scores, 0)     # scores[i] : scores of sub_traj with max_length i in  all_batch

        # all_scores is a tensor of shape (max_length * (max_length + 1) / 2, n_trajectories)
        n_rows = int(trajectories.max_length * (1 + trajectories.max_length) / 2)
        int(trajectories.max_length * ( trajectories.max_length-1) / 2)
        if self.weighing == "equal_within":
            # the following tensor represents the inverse of how many sub-trajectories there are in each trajectory
            contributions = 2.0 / (trajectories.when_is_done * (trajectories.when_is_done + 1))
            contributions = contributions / len(trajectories)
            # if we repeat the previous tensor, we get a tensor of shape (max_length * (max_length + 1) / 2, n_trajectories)
            # that we can multiply with all_scores to get a loss where each sub-trajectory is weighted equally within each trajectory
            contributions = contributions.repeat(n_rows, 1)
        elif self.weighing == "equal":
            n_sub_trajectories = int((trajectories.when_is_done * (trajectories.when_is_done + 1) / 2).sum().item())
            contributions = torch.ones(n_rows, len(trajectories),device=trajectories.env.device) / n_sub_trajectories
        elif self.weighing == "geometric_within":
            # the following tensor represents the weights given to each possible sub-trajectory length
            contributions = (self.lamda ** torch.arange(trajectories.max_length,device=trajectories.env.device).double()).float()
            contributions = contributions.unsqueeze(-1).repeat(1, len(trajectories))
            contributions = contributions.repeat_interleave(torch.arange(trajectories.max_length, 0, -1,device=trajectories.env.device),dim=0,
                                                            output_size=int(trajectories.max_length * (trajectories.max_length + 1) / 2))
            r"""
            Now we need to divide each column by n + (n-1) lambda +...+ 1*lambda^{n-1}
            where n is the length of the trajectory corresponding to that column
            We can do it the ugly way, or using the cool identity:
            https://www.wolframalpha.com/input?i=sum%28%28n-i%29+*+lambda+%5Ei%2C+i%3D0..n%29
            """
            per_trajectory_denominator = (1.0/ (1 - self.lamda) ** 2*
                                          (self.lamda * (self.lamda ** trajectories.when_is_done.double() - 1)
                                           + (1 - self.lamda) * trajectories.when_is_done.double())).float()
            contributions = contributions / per_trajectory_denominator
            contributions = contributions / len(trajectories)
        elif self.weighing == "geometric":
            # The position i of the following 1D tensor represents the number of sub-trajectories of length i in the batch
            # n_sub_trajectories = torch.maximum(
            #     trajectories.when_is_done - torch.arange(3).unsqueeze(-1),
            #     torch.tensor(0),
            # ).sum(1)
            # The following tensor's k-th entry represents the mean of all losses of sub-trajectories of length k
            per_length_losses = torch.stack([scores[~flattening_mask].pow(2).mean()
                                             for scores, flattening_mask in zip(scores, flattening_masks)])
            ld = self.lamda
            weights = ((1 - ld)/ (1 - ld**trajectories.max_length)*
                       (ld** torch.arange(trajectories.max_length, device=per_length_losses.device)))
            assert (weights.sum() - 1.0).abs() < 1e-5, f"{weights.sum()}"
            return (per_length_losses * weights).sum()
        elif self.weighing == "TB":
            indices = ( trajectories.max_length * (trajectories.when_is_done - 1)
                        - (trajectories.when_is_done - 2+1) * (trajectories.when_is_done - 2) / 2).long()
            #等差数列 减去多数的index  首项 1 末项 traj_lenth-2   一共traj_lenth-2 个
            return all_scores[indices,torch.arange(len(trajectories))].pow(2).mean()
        elif self.weighing == "DB":
            # Longer trajectories contribute more to the loss
            return scores[0][~flattening_masks[0]].pow(2).mean() # only consider trajtories with length 1

        elif self.weighing == "ModifiedDB":
            # The following tensor represents the inverse of how many transitions there are in each trajectory
            contributions = 1.0 / trajectories.when_is_done
            contributions = contributions / len(trajectories)
            contributions = contributions.repeat(trajectories.max_length, 1)
            contributions = torch.cat((contributions,torch.zeros((n_rows - trajectories.max_length,len(trajectories)),device=contributions.device)),0)
        else:
            raise ValueError(f"Unknown weighing method {self.weighing}")

        flat_contributions = contributions[~flattening_mask]
        assert (flat_contributions.sum() - 1.0).abs() < 1e-5, f"{flat_contributions.sum()}"
        losses = flat_contributions * all_scores[~flattening_mask].pow(2)
        return losses.sum()
