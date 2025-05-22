from abc import ABC, abstractmethod
from typing import ClassVar, Literal, Tuple, cast

import copy, pickle
import  os
import torch
from gymnasium.spaces import Discrete
from torchtyping import TensorType
from src.gfn.containers.states import States
from src.gfn.envs.env import Env
from src.gfn.envs.preprocessors import KHotPreprocessor
# Typing
TensorLong = TensorType["batch_shape", torch.long]
TensorFloat = TensorType["batch_shape", torch.float]
StatesTensor = TensorType["batch_shape", "state_shape", torch.float]
BatchTensor = TensorType["batch_shape"]
EnergyTensor = TensorType["state_shape", "state_shape", torch.float]
ForwardMasksTensor = TensorType["batch_shape", "n_actions", torch.bool]
BackwardMasksTensor = TensorType["batch_shape", "n_actions - 1", torch.bool]
from src.gfn.envs.bioseq import Oracle
from src.gfn.envs.bitseq import Replay_x,nbase2dec
class BioSeqPendEnv(Env):
    def __init__(
        self,
        ndim: int,
        oracle_path,
        mode_path=None,
        alpha:int = 3.0,
        R_max:float  =10.0,
        R_min:float  =1e-3,
        device_str: Literal["cpu", "cuda"] = "cpu",
        preprocessor_name: Literal["KHot"] = "KHot",
        nbase=4,
        name="TFbind8"
    ):
        self.ndim = ndim
        self.nbase = nbase
        s0 = torch.full((ndim,), -1, dtype=torch.long, device=torch.device(device_str))
        sf = torch.full((ndim,), nbase, dtype=torch.long, device=torch.device(device_str))
        self.oracle  = Oracle(nbase,ndim,
                              oracle_path,mode_path,
                              reward_exp=alpha,reward_max=R_max,reward_min=R_min,name=name)
        action_space = Discrete(2*nbase + 1)
        bction_space = Discrete(2)
        # the last action is the exit action that is only available for complete states
        # Action i  in [0, nbase - 1]  corresponds to preppend s with i,
        # Action i  in [nbase, 2base-1] corresponds tp append s with i
        if preprocessor_name == "KHot":
            preprocessor = KHotPreprocessor(height=nbase, ndim=ndim,fill_value=-1)  #heght=nbase+1 for encoding -1 or =nbase,fill_value =-1
        else:
            raise ValueError(f"Unknown preprocessor {preprocessor_name}")

        self.replay_x = Replay_x(nbase=nbase, ndim=ndim)
        super().__init__(action_space=action_space,
                         bction_space=bction_space,
                         s0=s0, sf=sf,
                         device_str=device_str,
                         preprocessor=preprocessor,)
    def make_States_class(self) -> type[States]:
        env = self
        class BioSeqStates(States):
            state_shape: ClassVar[tuple[int, ...]] = (env.ndim,)
            s0 = env.s0
            sf = env.sf
            action_base=env.nbase * torch.arange(0,env.ndim+1,1)
            @classmethod
            def make_random_states_tensor(
                cls, batch_shape: Tuple[int, ...]
            ) -> StatesTensor:
                states_tensor=torch.randint(-1, 2, batch_shape + (env.ndim,), dtype=torch.long, device=env.device)
                return  states_tensor

            def make_masks(self) -> Tuple[ForwardMasksTensor, BackwardMasksTensor]:
                "Mask illegal (forward and backward) actions."
                rep_dims = len(self.batch_shape) * (1,) + (2*env.nbase,)
                forward_masks = torch.ones((*self.batch_shape, env.n_actions),dtype=torch.bool,device=env.device)
                forward_masks[..., :-1] = (self.states_tensor==-1).any(-1,keepdim=True).repeat(rep_dims)
                forward_masks[..., -1]  = (self.states_tensor !=-1).all(-1)
                rep_dims = len(self.batch_shape) * (1,) + (2,)
                backward_masks          = (~self.is_initial_state).unsqueeze(-1).repeat(rep_dims)
                return forward_masks, backward_masks

            def update_masks(self,action=None,index=None) -> None:
                "Update the masks based on the current states."
                rep_dims = len(self.batch_shape) * (1,) + (2 * env.nbase,)
                self.forward_masks[...,:-1] = (self.states_tensor==-1).any(-1,keepdim=True).repeat(rep_dims)
                self.forward_masks[..., -1] = (self.states_tensor !=-1).all(-1)
                rep_dims = len(self.batch_shape) * (1,) + (2,)
                self.backward_masks          = (~self.is_initial_state).unsqueeze(-1).repeat(rep_dims)
        return BioSeqStates

    def bction2action(self,states: States, bctions: TensorLong) ->TensorLong:
        actions=torch.full_like(bctions,fill_value=-1)
        actions[bctions==0]=states.states_tensor[bctions==0,self.back_index(states.states_tensor[bctions==0])-1]
        actions[bctions==1]=self.nbase+states.states_tensor[bctions==1,0]
        return actions
    def action2bction(self,states: States, actions: TensorLong) ->TensorLong:
        bctions=torch.div(actions, self.nbase, rounding_mode='floor')
        return bctions

    def foward_index(self,states:StatesTensor ) -> BatchTensor:
        #argmin to find the last -1 element
        return (states > -1).int().argmin(-1)

    def back_index(self,states:StatesTensor ) -> BatchTensor:
        # can use argmax to find the last nbase element
        index=(states > -1).int().argmin(-1)
        index[index==0]=self.ndim
        return index

    def maskless_step(self, states: StatesTensor, actions: BatchTensor) -> StatesTensor:
        new_states = states.clone()
        sources= torch.div(actions, self.nbase, rounding_mode='floor')
        targets= torch.fmod(actions, self.nbase)
        new_states[sources ==0,self.foward_index(states[sources ==0])]     = targets[sources ==0]  #append
        new_states[sources ==1,1:] = states[sources ==1,:-1]                           # -1 element must be in the end of the sequence
        new_states[sources ==1,0]  = targets[sources ==1]                              #prepend
        return new_states

    def maskless_backward_step(self, states: StatesTensor, actions: BatchTensor) -> StatesTensor:
        new_states=states.clone()
        sources= torch.div(actions, self.nbase, rounding_mode='floor')
        new_states[sources ==0,self.back_index(states[sources ==0])-1]  = -1                       #append
        new_states[sources ==1,:-1] = states[sources ==1,1:]
        new_states[sources ==1,self.back_index(states[sources == 1])-1] = -1   #prepend   ???
        return new_states

    def log_reward(self, final_states: States) -> BatchTensor:
        raw_states = final_states.states_tensor
        return self.oracle(raw_states).log()
    @property
    def mean_reward(self)->torch.float:
        return (self.oracle.O_y **2).sum()/self.oracle.O_y.sum()
    @property
    def log_partition(self) -> torch.float:
        return (self.oracle.O_y.sum()).log()
