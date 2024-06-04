from abc import ABC, abstractmethod
from typing import ClassVar, Literal, Tuple, cast,List

import torch
#import torch.nn as nn
from gymnasium.spaces import Discrete
from torchtyping import TensorType
from functorch import vmap
from src.gfn.containers.states import States
from src.gfn.envs.env import Env
# Typing
TensorFloat = TensorType["batch_shape", torch.float]
StatesTensor = TensorType["batch_shape", "state_shape", torch.float]
BatchTensor = TensorType["batch_shape"]
ForwardMasksTensor = TensorType["batch_shape", "n_actions", torch.bool]
BackwardMasksTensor = TensorType["batch_shape", "n_actions - 1", torch.bool]

##################################
# bit-decimal transform
##################################
def dec2bin(x, length):
    #length: bits seqence
    canonical_base = 2 ** torch.arange(length - 1, -1, -1).to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(canonical_base).ne(0).float()

def nbase2dec(n,b, length):
    #n:n_base b:bits
    canonical_base = n ** torch.arange(length - 1, -1, -1).to(b.device, b.dtype)
    return torch.sum(canonical_base * b, -1)

class Replay_X:
    def __init__(
            self,ndim,capacity: int = int(1e6), ):
        self.ndim=ndim
        self.capacity = capacity
        self._is_full = False
        self._index = 0
        self.terminating_states = torch.LongTensor(0,ndim)
        self.terminating_rewards= torch.FloatTensor(0)
    def __repr__(self):
        return f"ReplayBuffer(capacity={self.capacity}, containing {len(self)} terminating states)"

    def __len__(self):
        return self.capacity if self._is_full else self._index

    def add(self, terminating_states:States,rewards:TensorFloat):
        to_add = len(terminating_states)
        self._is_full |= self._index + to_add >= self.capacity
        self._index = (self._index + to_add) % self.capacity
        #
        self.terminating_states=torch.cat(( self.terminating_states,terminating_states.states_tensor))
        self.terminating_rewards=torch.cat(( self.terminating_rewards,rewards))
        self.terminating_states =  self.terminating_states[-self.capacity:]
        self.terminating_rewards =  self.terminating_rewards[-self.capacity:]
        #
    def is_in_replay(self,states:States,valid_index):
        valid_masks= states.states_tensor!=-1
        state_index=torch.abs(valid_masks*
                              (states.states_tensor-
                               self.terminating_states[valid_index])).sum(-1)
        return state_index==0

    # def scores(self,state:States,nstate:States,valid_index):
    #     valid_index= torch.full_like(self.terminating_rewards, fill_value=True,dtype=torch.bool) \
    #         if valid_index is None else valid_index
    #     new_index=valid_index.clone()
    #     ##############################
    #     state_index=self.is_in_replay(state,valid_index)
    #     new_index[valid_index]=state_index
    #     nstate_index=self.is_in_replay(nstate,new_index)
    #     ##############################
    #     scores_Z=self.terminating_rewards[valid_index][state_index].sum() \
    #         if torch.any( state_index) else torch.tensor(1.)
    #     scores=self.terminating_rewards[new_index][nstate_index].sum() \
    #         if torch.any( nstate_index) else torch.tensor(0.)
    #     return scores,scores_Z,new_index

    def scores(self,state:States,nstate:States,state_list,valid_index):
        valid_index= torch.full_like(self.terminating_rewards, fill_value=True,dtype=torch.bool) \
            if valid_index is None else valid_index
        new_index=valid_index.clone()
        ##############################
        state_index=self.is_in_replay(state,valid_index)
        new_index[valid_index]=state_index
        nstate_index=[self.is_in_replay(s,new_index) for s in nstate[state_list]]
        ##############################
        scores=torch.zeros(nstate.batch_shape,dtype=torch.float)
        scores[state_list]=torch.stack([self.terminating_rewards[new_index][idx].mean()
                if torch.any( idx) else torch.tensor(0.) for idx in nstate_index])
        return scores,new_index

class Replay_x:
    def __init__(
            self,nbase,ndim,capacity: int = int(1e10), ):
        self.nbase=nbase
        self.ndim=ndim
        self.capacity = capacity
        self._is_full = False
        self._index = 0
        self.terminating_index = torch.LongTensor(0)
        self.terminating_rewards= torch.FloatTensor(0)
    def __repr__(self):
        return f"ReplayBuffer(capacity={self.capacity}, containing {len(self)} terminating states)"

    def __len__(self):
        return self.capacity if self._is_full else self._index

    def add(self, terminating_states:States,rewards:TensorFloat):
        to_add = len(terminating_states)
        self._is_full |= self._index + to_add >= self.capacity
        self._index = (self._index + to_add) % self.capacity
        #
        self.terminating_index=torch.cat(( self.terminating_index,
                                           nbase2dec(self.nbase,terminating_states.states_tensor.long(),self.ndim)))
        self.terminating_rewards=torch.cat((self.terminating_rewards,rewards))
        self.terminating_index =  self.terminating_index[-self.capacity:]
        self.terminating_rewards =  self.terminating_rewards[-self.capacity:]


class Oracle(ABC):
    def __init__(self, O_x: StatesTensor,alpha:float):
        super().__init__()
        self.alpha =alpha
        self.O_x = O_x#2*R-1
        hamming =lambda x, idx: torch.abs(idx * (x - self.O)).sum(-1).min(-1)[0]
        self.hamming=vmap(hamming)
    def __call__(self, states: StatesTensor) -> BatchTensor:
        valid_masks= states!=-1
        states = states.float()
        batch_shape,dim= states.shape[:-1],states.shape[-1]
        if states.ndim>2:
            dist=self.hamming(states.reshape(-1,dim), valid_masks(-1))
            return dist.reshape(batch_shape)
        else:
            dist=self.hamming(states, valid_masks)
            return -self.alpha * dist

class BitSeqEnv(Env):
    def __init__(
        self,
        ndim: int,
        alpha: float = 0.1,
        num_mode:int =2**4,
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
        self.nbase= 2
        self.num_mode= num_mode

        s0 = torch.full((ndim,), -1, dtype=torch.long, device=torch.device(device_str))
        sf = torch.full((ndim,), 2, dtype=torch.long, device=torch.device(device_str))
        mode_number=4
        step=torch.tensor(2**ndim-self.num_mode*mode_number).div(mode_number+1,rounding_mode="trunc")
        codebook=torch.concat([ torch.arange((i+1)*step,(i+1)*step+self.num_mode) for i in range(mode_number)])
        codebook=dec2bin(codebook,ndim)
        #codebook= torch.randint(0,2,(self.num_mode*mode_number,ndim))
        self.oracle = Oracle(codebook,alpha)

        action_space = Discrete(self.nbase * self.ndim + 1)
        # the last action is the exit action that is only available for complete states
        # Action i in [0, ndim - 1] corresponds to replacing s[i] with 0
        # Action i in [ndim, 2 * ndim - 1] corresponds to replacing s[i - ndim] with 1
        super().__init__(action_space=action_space, s0=s0, sf=sf,
                         device_str=device_str)

    def make_States_class(self) -> type[States]:
        env = self

        class BitSeqStates(States):
            state_shape: ClassVar[tuple[int, ...]] = (env.ndim,)
            s0 = env.s0
            sf = env.sf

            @classmethod
            def make_random_states_tensor(
                cls, batch_shape: Tuple[int, ...]
            ) -> StatesTensor:
                states_tensor=torch.randint(-1, env.nbase, batch_shape + (env.ndim,), dtype=torch.long, device=env.device)
                return  states_tensor

            def make_masks(self) -> Tuple[ForwardMasksTensor, BackwardMasksTensor]:
                forward_masks = torch.zeros(self.batch_shape + (env.n_actions,),device=env.device,dtype=torch.bool)
                rep_dims=len(self.batch_shape)*(1,)+(env.nbase,)
                forward_masks[..., :-1] = (self.states_tensor == -1).repeat(rep_dims)
                forward_masks[..., -1] =  (self.states_tensor != -1).all(dim=-1)
                #######################
                backward_masks = self.states_tensor.repeat(rep_dims) == \
                                 torch.arange(0, env.nbase, 1).repeat_interleave(env.ndim)
                return forward_masks, backward_masks # nbase =2

            def update_masks(self,action=None,index=None) -> None:
                rep_dims=len(self.batch_shape)*(1,)+(env.nbase,)
                self.forward_masks[...,:-1] = (self.states_tensor == -1).repeat(rep_dims)     # logit are empty we can filled iy  by [0,i].
                self.forward_masks[..., -1] = (self.states_tensor != -1).all(dim=-1)     #when all logits are filled, we can terminating the generating process by s_f
                #######
                self.backward_masks=self.states_tensor.repeat(rep_dims) == \
                                    torch.arange(0, env.nbase, 1).repeat_interleave(env.ndim)         # logit are filled by i=[0-nbase], we can take backward actions to remove i and denote the empty logit -1
        return BitSeqStates


    def maskless_step(self, states: StatesTensor, actions: BatchTensor) -> StatesTensor:
        targets= torch.div(actions, self.ndim, rounding_mode='floor') #  fill sources slot, 1....,ndim by target digit 1,....nbase
        sources=torch.fmod(actions, self.ndim)                        #  [digit 1: slot 1....,ndim,digit 2: slot 1..., ndim
        return  states.scatter_(-1, sources.unsqueeze(-1), targets.unsqueeze(-1))

    def maskless_backward_step(self, states: StatesTensor, actions: BatchTensor) -> StatesTensor:
        sources= torch.fmod(actions, self.ndim)                        #  sources: state element index
        return states.scatter_(-1, sources.unsqueeze(-1), -1)           #  target: -1


    def log_reward(self, final_states: States) -> BatchTensor:
        raw_states = final_states.states_tensor
        canonical =  raw_states#2 * raw_states - 1
        return self.oracle(canonical)

    def get_states_indices(self, states: States) -> BatchTensor:
        """The chosen encoding is the following: -1 -> 0, 0 -> 1, 1 -> 2,.... then we convert to base 5"""
        return nbase2dec(self.nbase+1, states.states_tensor+1,self.ndim).long().cpu().tolist()

    def get_terminating_states_indices(self, states: States) -> BatchTensor:
        return nbase2dec(self.nbase, states.states_tensor,self.ndim).long().cpu().tolist()

    @property
    def n_states(self) -> int:
        return (self.nbase+1)**self.ndim

    @property
    def n_terminating_states(self) -> int:
        return self.nbase**self.ndim

    @property
    def all_states(self) -> States:
        # This is brute force !
        digits = torch.arange(self.nbase+1, device=self.device)
        all_states = torch.cartesian_prod(*[digits] * self.ndim)# add -1 for s_f state
        all_states = all_states - 1
        return self.States(all_states)


    @property
    def ordered_states(self) -> States:
        # This is brute force !
        ordered_states = self.all_states
        index=[]
        for i in reversed(range(self.ndim+1)):
            index.append(torch.where(torch.sum(ordered_states.states_tensor==-1,-1) ==i)[0])
        return ordered_states[torch.cat(index)]

    @property
    def ordered_states_list(self) -> List[States]:
        # This is brute force !
        ordered_states = self.all_states
        index=[]
        for i in reversed(range(self.ndim+1)):
            index.append(torch.where(torch.sum(ordered_states.states_tensor==-1,-1) ==i)[0])
        return [ordered_states[ind] for ind in index]

    @property
    def terminating_states(self) -> States:
        digits = torch.arange(self.nbase, device=self.device)
        all_states = torch.cartesian_prod(*[digits] * self.ndim)
        return self.States(all_states)

    @property
    def true_dist_pmf(self) -> torch.Tensor:
        log_reward=self.log_reward(self.terminating_states)
        true_dist = log_reward-torch.logsumexp(log_reward,-1)
        return true_dist.exp().cpu()

    @property
    def log_partition(self) -> torch.float:
        log_rewards = self.log_reward(self.terminating_states)
        return torch.logsumexp(log_rewards, -1)

    @property
    def mean_reward(self)->torch.float:
        return (2 * self.log_reward(self.terminating_states)-self.log_partition).exp().sum()