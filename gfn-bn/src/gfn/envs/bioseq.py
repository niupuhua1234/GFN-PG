from abc import ABC, abstractmethod
from typing import ClassVar, Literal, Tuple, cast

import copy, pickle

import torch
from gymnasium.spaces import Discrete
from torchtyping import TensorType
from functorch import vmap
from src.gfn.containers.states import States
from src.gfn.envs.env import Env
from src.gfn.envs  import BitSeqEnv
from src.gfn.envs.preprocessors import KHotPreprocessor
import os
# Typing
TensorLong = TensorType["batch_shape", torch.long]
TensorFloat = TensorType["batch_shape", torch.float]
StatesTensor = TensorType["batch_shape", "state_shape", torch.float]
BatchTensor = TensorType["batch_shape"]
EnergyTensor = TensorType["state_shape", "state_shape", torch.float]
ForwardMasksTensor = TensorType["batch_shape", "n_actions", torch.bool]
BackwardMasksTensor = TensorType["batch_shape", "n_actions - 1", torch.bool]
from src.gfn.envs.bitseq import Replay_X,Replay_x,dec2bin,nbase2dec

# class PG_X:
#     def __init__(self,env:BitSeqEnv,inf_value=-1e5):
#         self.env=env
#         self.is_in = vmap(lambda x,idx,y: torch.abs(idx*(x-y)).sum(-1) == 0 ,in_dims=(0,0,None))
#         self.inf_value=torch.tensor(inf_value)
#         self.log_pgs=torch.zeros(size=self.env.all_states.forward_masks.shape)
#
#     def is_in_replay(self,states:States):
#         valid_masks= states.states_tensor!=-1
#         state_index=self.is_in(states.states_tensor,valid_masks,self.env.oracle.O_x)
#         return state_index
#
#     def mean_reward(self,states:States):
#         valid_index=~states.is_sink_state
#         mean_reward =torch.zeros(states.batch_shape,dtype=torch.float)
#         state_index=self.is_in_replay(states[valid_index])
#         mean_reward[valid_index]=torch.stack([torch.mean(self.env.oracle.O_y[idx]) for idx in state_index ])
#         return mean_reward
#
#     def __call__(self):
#         ordered_states_list = self.env.ordered_states_list
#         for order_idx, ordered_states in enumerate(ordered_states_list):  # filter out sf induced by terminating actions
#             print('order:',order_idx)
#             log_pgs = torch.zeros(size=ordered_states.forward_masks.shape)
#             for i,state in enumerate(ordered_states):
#                 print(i)
#                 next_states_all = self.env.all_step(state)
#                 scores           = self.mean_reward(next_states_all)
#                 log_pgs[i]         = (scores/scores.sum()).log().maximum(self.inf_value)
#             self.log_pgs[self.env.get_states_indices(ordered_states)]=log_pgs

class Oracle(ABC):
    def __init__(self, nbase,ndim,oracle_path,mode_path=None,reward_exp=3,reward_max=10.0,reward_min=1e-3,name="TFbind8"):
        super().__init__()
        print(f'Loading Oracle Data ...')
        with open(oracle_path, 'rb') as f:
            oracle_data = pickle.load(f)
        oracle_data['x']=torch.tensor(oracle_data['x'])

        if name=='TFbind10':
            from scipy.special import expit
            oracle_data['y']=  expit(oracle_data['y']* 3 )                                 #for tfbind10
        oracle_data['y'] = torch.tensor(oracle_data['y'], dtype=torch.float)
        oracle_data['y'] = torch.maximum(oracle_data['y'], torch.tensor(reward_min))# for qm9str sshe

        oracle_data['y'] =oracle_data['y']**reward_exp  # milden sharpness
        oracle_data['y']=oracle_data['y']* reward_max/oracle_data['y'].max() # scale up
        #oracle_data['y']=torch.maximum(oracle_data['y'],torch.tensor(reward_min)) # scale down  for tf8
        self.O_y = oracle_data['y'].squeeze()
        self.O_x =oracle_data['x']
        self.nbase=nbase
        self.ndim=ndim

        if mode_path is not None:
            with open(mode_path, 'rb') as f:
                modes  = pickle.load(f)
            self.modes = nbase2dec(nbase, torch.tensor(modes).long(), ndim)
        else:
            num_modes = int(len(self.O_y) * 0.001) if name=="sehstr" else int(len(self.O_y) * 0.005) # .005 for qm9str
            sorted_index = torch.sort(self.O_y)[1]
            self.modes   = sorted_index[-num_modes:]
        self.is_index_modes=torch.full_like(self.O_y, fill_value=False, dtype=torch.bool)
        self.is_index_modes[self.modes]=True

    def __call__(self, states: StatesTensor)-> BatchTensor:
        self.O_y.to(states.device)
        states = nbase2dec(self.nbase,states.long(),self.ndim)
        reward=self.O_y[states]
        return reward

class BioSeqEnv(BitSeqEnv,Env):
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
        """Discrete EBM environment.

        Args:
            ndim (int, optional): dimension D of the sampling space {0, 1,2,3}^D.
            energy (EnergyFunction): energy function of the EBM. Defaults to None. If None, the Ising model with Identity matrix is used.
            alpha (float, optional): scaling factor for oracle. Defaults to 1.0.
            device_str (str, optional): "cpu" or "cuda". Defaults to "cpu".
        """
        self.ndim = ndim
        self.nbase = nbase
        s0 = torch.full((ndim,), -1, dtype=torch.long, device=torch.device(device_str))   #  can use argmin to find the last -1 element
        sf = torch.full((ndim,), nbase, dtype=torch.long, device=torch.device(device_str)) # can use argmax to find the last nbase element
        self.oracle  = Oracle(nbase,ndim,
                              oracle_path,mode_path,
                              reward_exp=alpha,reward_max=R_max,reward_min=R_min,name=name)
        action_space = Discrete(nbase * ndim + 1)
        # the last action is the exit action that is only available for complete states
        # Action in [i*ndim, i*ndim - 1] (i in [0,nbase-1] corresponds to append s with (nbase-1),
        # i.e replace the last -1 in s with i
        if preprocessor_name == "KHot":
            preprocessor = KHotPreprocessor(height=nbase, ndim=ndim,fill_value=-1)
        else:
            raise ValueError(f"Unknown preprocessor {preprocessor_name}")

        self.replay_X=Replay_X(ndim=ndim)
        self.replay_x=Replay_x(nbase=nbase,ndim=ndim)
        Env.__init__(self,action_space=action_space,s0=s0, sf=sf,
                     device_str=device_str,preprocessor=preprocessor)

    def log_reward(self, final_states: States) -> BatchTensor:
        raw_states = final_states.states_tensor
        return self.oracle(raw_states).log()

    @property
    def mean_reward(self)->torch.float:
        return (self.oracle.O_y **2).sum()/self.oracle.O_y.sum()
    @property
    def log_partition(self) -> torch.float:
        return (self.oracle.O_y.sum()).log()