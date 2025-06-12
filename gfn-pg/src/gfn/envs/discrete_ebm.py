from abc import ABC, abstractmethod
from typing import ClassVar, Literal, Tuple, cast

import torch
import torch.nn as nn
from gymnasium.spaces import Discrete
from torchtyping import TensorType
from functorch import vmap
from src.gfn.containers.states import States
from src.gfn.envs.env import Env
# Typing
TensorFloat = TensorType["batch_shape", torch.float]
StatesTensor = TensorType["batch_shape", "state_shape", torch.float]
BatchTensor = TensorType["batch_shape"]
EnergyTensor = TensorType["state_shape", "state_shape", torch.float]
ForwardMasksTensor = TensorType["batch_shape", "n_actions", torch.bool]
BackwardMasksTensor = TensorType["batch_shape", "n_actions - 1", torch.bool]


class EnergyFunction(nn.Module, ABC):
    """Base class for energy functions"""
    @abstractmethod
    def forward(self, states: StatesTensor) -> BatchTensor:
        pass

class IsingModel(EnergyFunction):
    """Ising model energy function"""

    def __init__(self, J: EnergyTensor):
        super().__init__()
        self.J = J
        self.linear = nn.Linear(J.shape[0], 1, bias=False)
        self.linear.weight.data = J

    def forward(self, states: StatesTensor) -> BatchTensor:
        states = states.float()
        tmp = self.linear(states)
        return -(states.dot(tmp)).sum(-1)


class BitSet(EnergyFunction):
    """Ising model energy function"""
    def __init__(self, R: EnergyTensor):
        super().__init__()
        self.R = R#2*R-1
        hamming=lambda x: torch.abs(x - self.R).sum(-1).min(-1)[0]#/2.
        self.hamming=vmap(hamming)

    def forward(self, states: StatesTensor) -> BatchTensor:
        states = states.float()
        batch_shape,dim= states.shape[:-1],states.shape[-1]
        if states.ndim>2:
            dist=self.hamming(states.reshape(-1,dim))
            return dist.reshape(batch_shape)
        else:
            dist=self.hamming(states)
            return dist
class DiscreteEBMEnv(Env):
    """Environment for discrete energy-based models, based on https://arxiv.org/pdf/2202.01361.pdf"""

    def __init__(
        self,
        ndim: int,
        energy: EnergyFunction | None = None,
        alpha: float = 0.1,
        mode_number=4,
        device_str: Literal["cpu", "cuda"] = "cpu",
    ):
        """Discrete EBM environment.

        Args:
            ndim (int, optional): dimension D of the sampling space {0, 1}^D.
            energy (EnergyFunction): energy function of the EBM. Defaults to None. If None, the Ising model with Identity matrix is used.
            alpha (float, optional): interaction strength the EBM. Defaults to 1.0.
            device_str (str, optional): "cpu" or "cuda". Defaults to "cpu".
        """
        self.ndim = ndim
        self.num_code=2**4

        s0 = torch.full((ndim,), -1, dtype=torch.long, device=torch.device(device_str))
        sf = torch.full((ndim,), 2, dtype=torch.long, device=torch.device(device_str))
        if energy is None:
            step=torch.tensor(2**ndim-self.num_code*mode_number).div(mode_number+1,rounding_mode="trunc")
            codebook=torch.concat([ torch.arange((i+1)*step,(i+1)*step+self.num_code) for i in range(mode_number)])
            codebook=self.dec2bin(codebook,ndim)
            #codebook= torch.randint(0,2,(self.num_code*mode_number,ndim))
            energy  =BitSet(codebook)
        #energy = IsingModel(torch.ones((ndim, ndim), device=torch.device(device_str)))
        self.energy: EnergyFunction = energy
        self.alpha = alpha

        action_space = Discrete(2 * ndim + 1)
        # the last action is the exit action that is only available for complete states
        # Action i in [0, ndim - 1] corresponds to replacing s[i] with 0
        # Action i in [ndim, 2 * ndim - 1] corresponds to replacing s[i - ndim] with 1

        super().__init__(action_space=action_space, s0=s0, sf=sf)

    def bin2dec(self,b, bits):
        #b:batch,n_bits
        mask = 2 ** torch.arange(bits - 1, -1, -1).to(b.device, b.dtype)
        return torch.sum(mask * b, -1)

    def dec2bin(self,x, bits):
        #bit: bits seqence
        mask = 2 ** torch.arange(bits - 1, -1, -1).to(x.device, x.dtype)
        return x.unsqueeze(-1).bitwise_and(mask).ne(0).float()

    def make_States_class(self) -> type[States]:
        env = self

        class DiscreteEBMStates(States):
            state_shape: ClassVar[tuple[int, ...]] = (env.ndim,)
            s0 = env.s0
            sf = env.sf

            @classmethod
            def make_random_states_tensor(
                cls, batch_shape: Tuple[int, ...]
            ) -> StatesTensor:
                states_tensor=torch.randint(-1, 2, batch_shape + (env.ndim,), dtype=torch.long, device=env.device)
                return  states_tensor

            def make_masks(self) -> Tuple[ForwardMasksTensor, BackwardMasksTensor]:
                forward_masks = torch.zeros(self.batch_shape + (env.n_actions,),device=env.device,dtype=torch.bool)
                forward_masks[..., : env.ndim] = (self.states_tensor == -1)
                forward_masks[..., env.ndim : 2 * env.ndim] = (self.states_tensor == -1)
                forward_masks[..., -1] = torch.all(self.states_tensor != -1, dim=-1)
                #######################
                backward_masks = torch.zeros(self.batch_shape + (env.n_actions - 1,),device=env.device,dtype=torch.bool)
                backward_masks[..., : env.ndim] = self.states_tensor == 0
                backward_masks[..., env.ndim : 2 * env.ndim] = (self.states_tensor == 1)

                return forward_masks, backward_masks

            def update_masks(self,action=None,index=None) -> None:
                self.forward_masks[..., : env.ndim] = (self.states_tensor == -1)              # logit ar eempty we can filled iy  by 0.
                self.forward_masks[..., env.ndim : 2 * env.ndim] = (self.states_tensor == -1) # logit are empty we can filled it by 1.
                self.forward_masks[..., -1] = torch.all(self.states_tensor != -1, dim=-1)     #when all logits are filled, we can terminating the generating process by s_f
                #######
                self.backward_masks[..., : env.ndim] = self.states_tensor == 0     # logit are filled by 0, we can take backward actions to remove 0 and denote the empty logit -1
                self.backward_masks[..., env.ndim : 2 * env.ndim] = (self.states_tensor == 1) # logit are filled by 1, we can take backward actions to remove 1 and denote the empty logit -1
        return DiscreteEBMStates

    def is_exit_actions(self, actions: BatchTensor) -> BatchTensor:
        return actions == self.n_actions - 1

    def maskless_step(self, states: StatesTensor, actions: BatchTensor) -> None:
        # First, we select that actions that replace a -1 with a 0
        mask_0 = actions < self.ndim
        states[mask_0] = states[mask_0].scatter(-1, actions[mask_0].unsqueeze(-1), 0)
        # Then, we select that actions that replace a -1 with a 1
        mask_1 = (actions >= self.ndim) & (actions < 2 * self.ndim)
        states[mask_1] = states[mask_1].scatter(-1, (actions[mask_1] - self.ndim).unsqueeze(-1), 1)
        return states

    def maskless_backward_step(self, states: StatesTensor, actions: BatchTensor) -> None:
        return states.scatter_(-1, actions.unsqueeze(-1).fmod(self.ndim), -1)

    def log_reward(self, final_states: States) -> BatchTensor:
        raw_states = final_states.states_tensor
        canonical = raw_states#2 * raw_states - 1
        return -self.alpha * self.energy(canonical)

    def get_states_indices(self, states: States) -> BatchTensor:
        """The chosen encoding is the following: -1 -> 0, 0 -> 1, 1 -> 2, then we convert to base 3"""
        states_raw = states.states_tensor
        canonical_base = 3 ** torch.arange(self.ndim - 1, -1, -1, device=self.device)
        return (states_raw + 1).mul(canonical_base).sum(-1).long().cpu().tolist()

    def get_terminating_states_indices(self, states: States) -> BatchTensor:
        states_raw = states.states_tensor
        canonical_base = 2 ** torch.arange(self.ndim - 1, -1, -1, device=self.device)
        return (states_raw).mul(canonical_base).sum(-1).long().cpu().tolist()

    @property
    def n_states(self) -> int:
        return 3**self.ndim

    @property
    def n_terminating_states(self) -> int:
        return 2**self.ndim

    @property
    def all_states(self) -> States:
        # This is brute force !
        digits = torch.arange(3, device=self.device)
        all_states = torch.cartesian_prod(*[digits] * self.ndim)
        all_states = all_states - 1
        return self.States(all_states)

    @property
    def terminating_states(self) -> States:
        digits = torch.arange(2, device=self.device)
        all_states = torch.cartesian_prod(*[digits] * self.ndim)
        return self.States(all_states)

    @property
    def true_dist_pmf(self) -> torch.Tensor:
        true_dist = torch.exp(self.log_reward(self.terminating_states))
        return true_dist / true_dist.sum()

    @property
    def true_marg_pmf(self) -> torch.Tensor:
        true_dist = torch.exp(self.log_reward(self.terminating_states))
        return true_dist / true_dist.sum()

    @property
    def log_partition(self) -> torch.float:
        log_rewards = self.log_reward(self.terminating_states)
        return torch.logsumexp(log_rewards, -1)