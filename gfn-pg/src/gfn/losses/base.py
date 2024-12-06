import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple
import torch
from torchtyping import TensorType

from src.gfn.containers.states import States
from src.gfn.containers.trajectories import Trajectories
from src.gfn.containers.transitions import Transitions
from src.gfn.estimators import LogitPBEstimator, LogitPFEstimator,LogEdgeFlowEstimator
LogPTrajectoriesTensor = TensorType["max_length", "n_trajectories", float]
####################
class Parametrization(ABC):
    """
    Abstract Base Class for Flow Parametrizations,
    as defined in Sec. 3 of GFlowNets Foundations.
    All attributes should be estimators, and should either have a GFNModule or attribute called `module`,
    or torch.Tensor attribute called `tensor` with requires_grad=True.
    """
    @property
    def parameters(self) -> dict:
        """
        Return a dictionary of all parameters of the parametrization.
        Note that there might be duplicate parameters (e.g. when two NNs share parameters),
        in which case the optimizer should take as input set(self.parameters.values()).
        """
        # TODO: use parameters of the fields instead, loop through them here
        parameters_dict = {}
        for name,estimator in self.__dict__.items():
            parameters_dict.update({name:estimator.parameters()})
        return parameters_dict

    def save_state_dict(self, path: str,index:str):
        for name, estimator in self.__dict__.items():
            torch.save(estimator.named_parameters(), os.path.join(path, index+name + ".pt"))

    def load_state_dict(self, path: str,index:str):
        for name, estimator in self.__dict__.items():
            estimator.load_state_dict(torch.load(os.path.join(path, index+name + ".pt")))

class FParametrization(Parametrization):
    def __init__(self, logF: LogEdgeFlowEstimator):
        self.logF=logF

class PFBasedParametrization(Parametrization, ABC):
    r"Base class for parametrizations that explicitly used :math:`P_F`"
    def __init__(self, logit_PF: LogitPFEstimator,logit_PB: LogitPBEstimator):
        self.logit_PF=logit_PF
        self.logit_PB=logit_PB
####################################
# loss objects                     #
# ##################################

class Loss(ABC):
    "Abstract Base Class for all GFN Losses"
    def __init__(self, parametrization):
        self.parametrization=parametrization
    @abstractmethod
    def __call__(self, *args, **kwargs) -> TensorType[0, float]:
        pass
class StateDecomposableLoss(Loss,ABC):
    def __init__(self, parametrization: FParametrization,fill_value=0.0):
        self.fill_value = fill_value
        super().__init__(parametrization)
    @abstractmethod
    def __call__(self, states_tuple: Tuple[States, States]) -> TensorType[0, float]:
        pass
    def edge_flow(self,states: States,actions):
        flow_all = self.parametrization.logF(states)
        flow_edge=torch.gather(flow_all, dim=-1, index=actions.unsqueeze(-1)).squeeze(-1)
        return   flow_edge

class Sub_TrajectoryDecomposableLoss(Loss,ABC):
    """
    Args:
        fill_value (float, optional):  LogP Value to use for invalid states (i.e. s_f that is added to shorter trajectories). Defaults to 0.0.
                                       Here we used 0.0 instead of inf_value to ensure stability.
        inf_value (float, optional):   LogP Value to use for zero probability.                      Defaults to -1e5 ( or -float('inf')).
        temperature (float, optional): Temperature to use for the softmax(correspond to how the actions_sampler evaluates each action.). Defaults to 1.0.
    """
    def __init__(self, parametrization: PFBasedParametrization,
                 fill_value=0.0,## only inf &0 can avoid masked log_value increase after summation of cumlog_prob
                 temperature=1.0,
                 inf_value=-1e5,): # not -inf for graidents stability
        self.fill_value =fill_value
        self.temperature=temperature
        self.inf_value:float=inf_value
        super().__init__(parametrization)

    @abstractmethod
    def __call__(self, states_tuple: Tuple[States, States]) -> TensorType[0, float]:
        pass
    @staticmethod
    def action_prob_gather(log_ps,actions):
        return torch.gather(log_ps, dim=-1, index=actions.unsqueeze(-1)).squeeze(-1)

    def forward_log_prob(self,states: States):
        logits=self.parametrization.logit_PF(states)
        if torch.any(torch.all(torch.isnan(logits), 1)):
            raise ValueError("NaNs in estimator")
        logits[~states.forward_masks] = self.inf_value            # note we use self.inf_value rathern self.fill_value
        log_all = (logits/ self.temperature).log_softmax(dim=-1)  # By log_softmax inf_value in logits can be recovered
        return log_all

    def backward_log_prob(self,states: States):
        logits=self.parametrization.logit_PB(states)
        if torch.any(torch.all(torch.isnan(logits), 1)): raise ValueError("NaNs in estimator")
        logits[~states.backward_masks] = self.inf_value
        log_all = logits.log_softmax(dim=-1)
        return log_all

class EdgeDecomposableLoss(Sub_TrajectoryDecomposableLoss,ABC):
    @abstractmethod
    def __call__(self, edges: Transitions) -> TensorType[0, float]:
        pass

class TrajectoryDecomposableLoss(Sub_TrajectoryDecomposableLoss,ABC):
    @abstractmethod
    def __call__(self, trajectories: Trajectories) -> TensorType[0, float]:
        pass

    @staticmethod
    def forward_state_actions(trajectories: Trajectories):
        """
        1. compute forward_prob for forward trajectory.
        Forward trajectory:             s0   ->    s1   ->..... ->  sT-1  ->s_T   (->sf)
                                        a0   ->    a1   ->....  ->  aT-1  ->a_T
        compute forward probability for:π|s0  ->  π|s1   ->..... ->π|sT-1 ->π|s_T
        2. compute forward_prob for backward trajectory. (used for backward guided policy)
        Backward trajectory:                          sT    ->    sT-1 -> ....  ->  s1 (-> s0)
                                                      aT-1   ->    aT-2 ->   ....->  a0
        compute forward probability for state: (ST)-> π|sT-1 ->   π|sT-2->...... ->π|s0 (错一格 和a 对齐)

        """

        if not(trajectories.is_backward):
            valid_index = ~trajectories.is_sink_action            # filtered padding s_f
            valid_states= trajectories.states[:-1][valid_index]   #s0->sf: [:-1] states traj is one bigger than actions traj by an s_f in forward samples
            valid_actions = trajectories.actions[valid_index]
        else:
            valid_index = ~trajectories.is_sink_action
            valid_states= trajectories.states[1:][valid_index]    #sT<-s0: [1:] states traj is one bigger than actions traj by an s_T in  backward samples
            valid_actions=trajectories.env.bction2action(trajectories.states[:-1][valid_index],  #[1:-1]?
                                                         trajectories.actions[valid_index])
        if valid_states.batch_shape != tuple(valid_actions.shape):
            raise AssertionError("Something wrong happening with log_pf evaluations")
        return valid_states,valid_actions,valid_index
    @staticmethod
    def backward_state_actions(trajectories: Trajectories):
        """
        1. compute backward_prob for forward trajectory.
        Forward trajectory:                       s0   ->    s1   ->..... ->  sT-1  (->sT ->sf )
                                                  a0   ->    a1   ->....  ->  aT-1  (->aT)
        compute back probability for: (s0)  ->  π|s1   ->.....  ->π|sT-1  ->  π|sT             (s错一格,和a 对齐)
        2. compute backward_prob for backward trajectory. (used for backward guided policy)
        Backward trajectory:            sT  ->    sT-1 ->...... ->  s1   (->s_0 )
                                      aT-1  ->    aT-2 ->.......->  a0
        compute back probability for: π|sT  ->  π|sT-1 ->...... ->π|s1
        """
        if not (trajectories.is_backward):
            inter_index=~trajectories.is_sink_action & ~trajectories.is_terminating_action # filter padding s_f and s_T
            non_init_valid_states  =  trajectories.states[1:][inter_index]                 # [1:] filter out s_0 and align with aciton(错一格)
            non_exit_valid_actions = trajectories.env.action2bction( trajectories.states[:-1][inter_index],
                                                                     trajectories.actions[inter_index])
        else:
            inter_index = ~trajectories.is_sink_action                     # filtered padding s_f( dummy states after reaching s0)
            non_init_valid_states = trajectories.states[:-1][inter_index]  #sT->s0 [:-1] states traj is one bigger than actions traj by an s_0
            non_exit_valid_actions = trajectories.actions[inter_index]
        return non_init_valid_states,non_exit_valid_actions,inter_index

    def cumulative_logprobs(
        self,
        trajectories: Trajectories,
        log_p_trajectories: LogPTrajectoriesTensor,
    ):
        """
        Args:
             trajectories: trajectories
             log_p_trajectories: log probabilities of each transition in each trajectory
        Return:
            cumulative sum of log probabilities of each trajectory
        """
        return torch.cat((torch.zeros(1, trajectories.n_trajectories, device=log_p_trajectories.device),
                          log_p_trajectories.cumsum(dim=0),),dim=0)

    def get_pfs(
        self,
        trajectories: Trajectories) -> LogPTrajectoriesTensor:
        """Evaluate log_pf for each action in each trajectory in the batch.
        Args:
            trajectories (Trajectories): Trajectories to evaluate.
            fill_value   (float)       : Values used for invalid states (sink state usually)
        Returns:
            Tuple[LogPTrajectoriesTensor | None, LogPTrajectoriesTensor]: A tuple of float tensors of shape (max_length, n_trajectories) containing the log_pf and log_pb for each action in each trajectory. The first one can be None.
        """
        valid_states, valid_actions,valid_index=self.forward_state_actions(trajectories)
        valid_log_pf_all=self.forward_log_prob(valid_states)
        valid_log_pf_actions = self.action_prob_gather(valid_log_pf_all,valid_actions)
        #assert torch.all((trajectories.log_probs[trajectories.actions != -1] - valid_log_pf_actions).abs() < 1e-3) if on policy
        log_pf_trajectories = torch.full_like(trajectories.actions, fill_value=self.fill_value, dtype=torch.float)
        log_pf_trajectories[valid_index] = valid_log_pf_actions
        #log_pf_trajectories_all = torch.full_like(trajectories.states[:-1].forward_masks, fill_value=self.fill_value, dtype=torch.float)
        #log_pf_trajectories_all[valid_index,:] = valid_log_pf_all
        return log_pf_trajectories

    def get_pbs(
            self,
            trajectories: Trajectories) -> LogPTrajectoriesTensor:
        """Evaluate log_pb for each action in each trajectory in the batch."""
        non_init_valid_states,non_exit_valid_actions,inter_index=self.backward_state_actions(trajectories)
        valid_log_pb_all=self.backward_log_prob(non_init_valid_states)
        valid_log_pb_actions=self.action_prob_gather(valid_log_pb_all,non_exit_valid_actions)
        #torch.all((trajectories.log_probs[trajectories.actions != -1][valid_actions != trajectories.env.n_actions - 1] - valid_log_pb_actions).abs() <= 1e-4)
        log_pb_trajectories = torch.full_like(trajectories.actions, fill_value=self.fill_value, dtype=torch.float)
        log_pb_trajectories[inter_index] = valid_log_pb_actions
        #log_pb_trajectories_all = torch.full_like(trajectories.states[:-1].backward_masks, fill_value=self.fill_value, dtype=torch.float)
        #log_pb_trajectories_all[inter_index] = valid_log_pb_all
        return log_pb_trajectories




