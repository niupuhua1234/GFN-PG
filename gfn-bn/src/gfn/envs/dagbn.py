"""
Copied and Adapted from https://github.com/Tikquuss/GflowNets_Tutorial
"""

from typing import ClassVar, Literal, Tuple, Union,Optional
import torch
from gymnasium.spaces import Discrete
from torchtyping import TensorType

from src.gfn.containers.states import States
from src.gfn.envs.env import Env
from src.gfn.envs.preprocessors import DictPreprocessor,IdentityPreprocessor
import numpy as np
from functorch import vmap
# Typing
TensorLong = TensorType["batch_shape", torch.long]
TensorFloat = TensorType["batch_shape", torch.float]
TensorBool = TensorType["batch_shape", torch.bool]
ForwardMasksTensor = TensorType["batch_shape", "n_actions", torch.bool]
BackwardMasksTensor = TensorType["batch_shape", "n_actions - 1", torch.bool]
OneStateTensor = TensorType["state_shape", torch.float]
StatesTensor = TensorType["batch_shape", "state_shape", torch.float]
NonValidActionsError = type("NonValidActionsError", (ValueError,), {})

class DAG_BN(Env):
    def __init__(
            self,
            n_dim,
            all_graphs,
            max_parents=None,
            device_str: Literal["cpu", "cuda"] = "cpu",
            score=None,
            alpha=1.
    ):
        """GFlowNet environment for learning a distribution over DAGs.

        Parameters
        ----------
        scorer : BaseScore instance
            The score to use. Note that this contains the data_bio.

        n_dim :  int, optional
            num_of variables.

        max_parents : int, optional
            Maximum number of parents for each node in the DAG. If None, then
            there is no constraint on the maximum number of parents.
        States: adjency matrix
        """
        self.alpha=alpha
        self.n_dim=n_dim
        self.max_parents = max_parents or self.n_dim# min( , )
        self.max_edges = self.n_dim * (self.n_dim - 1) // 2 # dag edge limite
        self.zero_score=score(torch.zeros(self.n_dim**2))
        self.score=score
        self.all_graphs=all_graphs
        self.all_indices={np.array2string(item,separator=','): idx for idx, item in enumerate(all_graphs.numpy())}
        #find_index=lambda x: torch.where(torch.all(x==self.all_graphs,-1))[0]
        #self.find_index=vmap(find_index)
        preprocessor = IdentityPreprocessor(output_shape=(n_dim**2,))  #DictPreprocessor(n_dim=n_dim,embed_dim=embed_dim)
        action_space = Discrete(self.n_dim ** 2 + 1)                   # all possible edges+stop action

        s0 = torch.zeros((n_dim*n_dim,), dtype=torch.long, device=torch.device(device_str))
        sf = torch.full( (n_dim*n_dim,), fill_value=-1, dtype=torch.long, device=torch.device(device_str))

        super().__init__(
            action_space=action_space,
            s0=s0,
            sf=sf,
            device_str=device_str,
            preprocessor=preprocessor,
        )

    def make_States_class(self) -> type[States]:
        "Creates a States class for this environment"
        env = self
        class DAG_States(States):

            state_shape: ClassVar[tuple[int, ...]] = (env.n_dim*env.n_dim,)
            s0:ClassVar[OneStateTensor]            = env.s0
            sf:ClassVar[OneStateTensor]            = env.sf

            def __init__(
                    self,
                    states_tensor: StatesTensor,
                    forward_masks: ForwardMasksTensor | None = None,
                    backward_masks: BackwardMasksTensor | None = None,
                    forward_closure_T: ForwardMasksTensor | None = None,
            ):
                super().__init__(states_tensor,
                                 forward_masks,
                                 backward_masks)

                if forward_closure_T is None:
                    self.forward_closure_T=(1-self.forward_masks[...,:-1].int()-self.states_tensor).bool()
                else:
                    self.forward_closure_T=forward_closure_T

            @classmethod
            def make_random_states_tensor(cls, batch_shape: Tuple[int, ...]) -> StatesTensor:
                "Creates a batch of random states."
                states_tensor = torch.randint(0, 2, batch_shape + env.s0.shape, device=env.device)
                return states_tensor

            def closure_T_exact(self,adjacency):
                reach=adjacency.reshape(*adjacency.shape[0:-1], env.n_dim, env.n_dim).transpose(-1,-2).bool()
                for k in range(env.n_dim):
                    for i in range(env.n_dim):
                        for j in range(env.n_dim):
                            reach[...,i,j]= reach[...,i,j] | ( reach[...,i,k] & reach[...,k,j])
                return (reach | torch.eye(env.n_dim, dtype=torch.bool)).reshape(*adjacency.shape[0:-1],env.n_dim**2)

            def make_masks(self) -> Tuple[ForwardMasksTensor, BackwardMasksTensor]:
                "Mask illegal (forward and backward) actions."
                #backward_masks = torch.ones((*self.batch_shape, env.n_actions-1), dtype=torch.bool, device=env.device)
                #backward_masks[...,[n * env.n_dim + n for n in range(env.n_dim)]]=False
                backward_masks=self.states_tensor.bool()
                forward_masks = torch.ones((*self.batch_shape, env.n_actions), dtype=torch.bool, device=env.device)
                new_masks=1-(self.states_tensor+self.closure_T_exact(self.states_tensor))
                forward_masks[...,:-1]=new_masks.bool()
                return forward_masks,backward_masks

            #forward_masks[..., :-1].reshape(*self.batch_shape, 11, 11)[5].int()

            def update_masks(self,actions:TensorLong,index=TensorLong):
                "Update the masks based on the current states."
                sources, targets = torch.div(actions[index], env.n_dim, rounding_mode='floor'), \
                                   torch.fmod(actions[index], env.n_dim)
                adjcency=self.states_tensor.reshape(*self.batch_shape, env.n_dim, env.n_dim)[index]
                old_closure_T=self.forward_closure_T.reshape(*self.batch_shape, env.n_dim, env.n_dim)[index]

                source_rows = old_closure_T[torch.arange(index.sum()), sources, :][...,None,:]  # insert one dim  [num_env,1,num_variables]
                target_cols = old_closure_T[torch.arange(index.sum()), :, targets][...,:,None]
                new_closure_T   =  torch.logical_or(old_closure_T, source_rows.logical_and(target_cols))
                # Update the masks (maximum number of parents)
                num_parents = torch.sum(adjcency, dim=-2, keepdim=True)
                # Update the masks
                new_masks=1- (new_closure_T+adjcency)
                new_masks=new_masks.mul(num_parents <= env.max_parents)
                #assert torch.all( (1-(self.states_tensor[index] + self.closure_T_exact(self.states_tensor[index]))) == new_masks.reshape(-1,env.n_dim**2))
                #print(torch.all(self.forward_masks[index][...,:-1]==self.forward_masks[index,:-1]))
                self.forward_masks[...,:-1][index]= new_masks.reshape(-1,env.n_dim**2).bool()
                self.forward_closure_T[index]=new_closure_T.reshape(-1, env.n_dim**2)
                self.backward_masks = self.states_tensor.bool()
        return DAG_States

    def maskless_step(self, states: StatesTensor,actions:TensorLong) -> StatesTensor:
        index = (torch.arange(0,  actions.shape[0]))
        states[...,index,actions]= 1
        return states

    def maskless_backward_step(self, states: StatesTensor,actions:TensorLong) -> StatesTensor:
        index = (torch.arange(0,  actions.shape[0]))
        states[...,index,actions]= 0
        return states

    def get_states_indices(self, states: States):
        #all_states={item:idx for idx,item,_ in enumerate(self.all_states(n_edges=self.max_edges))}
        indices    = [self.all_indices.get(np.array2string(i,separator=',')) for i in states.states_tensor.numpy()]
        return indices

    def get_terminating_states_indices(self, states: States):
        return self.get_states_indices(states)
    @ property
    def all_states(self):
        return self.States(self.all_graphs)
    @property
    def terminating_states(self) -> States:
        return self.States(self.all_graphs)
    @property
    def n_terminating_states(self) -> int:
        assert self.n_dim<=5
        return [0, 1, 3, 25, 543, 29281, 3781503][self.n_dim]

    @property
    def true_dist_pmf(self) -> torch.Tensor:
        log_reward=self.log_reward(self.terminating_states)
        true_dist = log_reward-torch.logsumexp(log_reward,-1)
        return true_dist.exp().cpu()#,idx_list

    def true_ratio_pmf(self,states: States) -> Tuple[torch.Tensor,list]:
        uni_states=torch.unique(states.states_tensor, dim=0)
        idx_list=self.empirical_all_states(states)
        log_reward=self.log_reward(self.States(uni_states))
        true_dist =  log_reward-torch.logsumexp(log_reward,-1)
        return true_dist.exp(),idx_list

    def log_reward(self, final_states:States):
        reward = self.score(final_states.states_tensor)-self.zero_score
        return reward/self.alpha
    @property
    def log_partition(self) -> torch.float:
        log_rewards = self.log_reward(self.terminating_states)
        return torch.logsumexp(log_rewards, -1)
    @property
    def mean_log_reward(self)->torch.float:
        return (self.true_dist_pmf * self.log_reward(self.terminating_states)).sum()

    def empirical_all_states(self, states: States):
        graph=[]
        uni_states=torch.unique(states.states_tensor, dim=0)
        for graph_edges in uni_states:
            graph.append(np.array2string(graph_edges.numpy(),separator=','))
        return list(set(graph))

    def empirical_states_indices(self,states:States):
        all_states= {item:idx for idx,item in enumerate(self.empirical_all_states(states))}
        indices   = [all_states.get(np.array2string(i,separator=',')) for i in states.states_tensor.numpy()]
        return indices

    def empirical_n_terminating_states(self,states:States):
        return len(self.empirical_all_states(states))

