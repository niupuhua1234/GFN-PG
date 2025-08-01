from abc import ABC, abstractmethod
from typing import ClassVar, Literal, Tuple, cast

import copy, pickle

import torch
from gymnasium.spaces import Discrete
from torchtyping import TensorType
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
ForwardMasksTensor = TensorType["batch_shape", "n_actions", torch.bool]
BackwardMasksTensor = TensorType["batch_shape", "n_actions - 1", torch.bool]
from src.gfn.envs.bitseq import nbase2dec

class Replay_G(ABC):
    def __init__(self, nbase, ndim, capacity: int = int(1e6), ):
        self.nbase    = nbase
        self.ndim     = ndim
        self.capacity = capacity
        self._is_full = False
        self._index = 0
        self.x_states = torch.LongTensor(0, ndim)
        self.x_rewards= torch.FloatTensor(0)

    def add(self, x_states: States, rewards: TensorFloat):
        to_add = len(x_states)
        self._is_full |= self._index + to_add >= self.capacity
        self._index = (self._index + to_add) % self.capacity
        #
        self.x_states = torch.cat((self.x_states, x_states.states_tensor))
        self.x_rewards= torch.cat((self.x_rewards, rewards))
        self.x_states = self.x_states[-self.capacity:]
        self.x_rewards= self.x_rewards[-self.capacity:]
        #
    def is_in_replay(self, states: States, valid_index):
        valid_masks= states.states_tensor!=-1
        state_index=torch.abs(valid_masks*
                              (states.states_tensor-
                               self.x_states[valid_index])).sum(-1)
        return state_index==0
    def scores_approx(self,state:States,nstate:States,valid_index):
        valid_index= torch.full_like(self.x_rewards, fill_value=True,dtype=torch.bool) \
            if valid_index is None else valid_index
        new_index=valid_index.clone()
        ##############################
        state_index=self.is_in_replay(state,valid_index)
        new_index[valid_index]=state_index
        nstate_index=self.is_in_replay(nstate,new_index)
        ##############################
        scores_Z=self.x_rewards[valid_index][state_index].sum() \
            if torch.any(state_index) else torch.tensor(1.)
        scores=self.x_rewards[new_index][nstate_index].sum() \
            if torch.any(nstate_index) else torch.tensor(0.)
        return scores,scores_Z,new_index
    def scores(self,state:States,nstate:States,state_list,valid_index):
        valid_index= torch.full_like(self.x_rewards, fill_value=True,dtype=torch.bool) \
            if valid_index is None else valid_index
        new_index=valid_index.clone()
        ##############################
        state_index=self.is_in_replay(state,valid_index)
        new_index[valid_index]=state_index
        nstate_index=[self.is_in_replay(s,new_index) for s in nstate[state_list]]
        ##############################
        scores=torch.zeros(nstate.batch_shape,dtype=torch.float)
        scores[state_list]=torch.stack([self.x_rewards[new_index][idx].mean()
                                        if torch.any(idx) else torch.tensor(0.) for idx in nstate_index])
        return scores,new_index
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
            self.modes = torch.tensor(modes).long()
        else:
            num_modes    = int(len(self.O_y) * 0.001) if name == "sehstr" else int(len(self.O_y) * 0.005) # .005 for qm9str
            sorted_index = torch.sort(self.O_y)[1]
            self.modes   = self.O_x[sorted_index[-num_modes:]]

    def is_mode(self, states: StatesTensor)-> BatchTensor:
        modes    = nbase2dec(self.nbase, self.modes, self.ndim)
        states   = nbase2dec(self.nbase,states.long(), self.ndim)
        matched  = torch.isin(states,modes)
        return  matched

    def __call__(self, states: StatesTensor)-> BatchTensor:
        states = nbase2dec(self.nbase,states.long(),self.ndim)
        reward = self.O_y.to(states.device)[states]
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
        """Discrete Graded Sequence environment.

        Args:
            nbase(int, optional): number   N of  integer set  {0,1,.....N}
            ndim (int, optional): dimension D of the sampling space {0,1,...,N}^D.
            alpha (float, optional): scaling factor for oracle. Defaults to 1.0.
            device_str (str, optional): "cpu" or "cuda". Defaults to "cpu".
        """
        self.ndim = ndim
        self.nbase = nbase
        s0 = torch.full((ndim,), -1, dtype=torch.long, device=torch.device(device_str))
        sf = torch.full((ndim,), nbase, dtype=torch.long, device=torch.device(device_str))
        self.oracle  = Oracle(nbase,ndim,
                              oracle_path,mode_path,
                              reward_exp=alpha,reward_max=R_max,reward_min=R_min,name=name)
        action_space = Discrete(nbase * ndim + 1)
        # the last action is the exit action that is only available for complete states
        # Action i in [0, ndim - 1] corresponds to replacing s[i] with 0
        # Action i in [ndim, 2*ndim - 1] corresponds to replacing s[i] with 1
        # Action i in [2*ndim, 3*ndim - 1] corresponds to replacing s[i] with 2
        # .......
        # Action i in [(nbase-1)*ndim, nbase * ndim - 1] corresponds to replacing s[i] with nbase-1

        if preprocessor_name == "KHot":
            preprocessor = KHotPreprocessor(height=nbase, ndim=ndim,fill_value=-1)
        else:
            raise ValueError(f"Unknown preprocessor {preprocessor_name}")

        self.replay_G=Replay_G(nbase=nbase,ndim=ndim)
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