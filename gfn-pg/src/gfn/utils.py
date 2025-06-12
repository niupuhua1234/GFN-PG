from typing import Dict, Optional

import torch

from src.gfn.containers import States, Trajectories, Transitions
from src.gfn.samplers import TrajectoriesSampler,DiscreteActionsSampler
from src.gfn.envs import Env,HyperGrid,DAG_BN,BitSeqEnv,BioSeqEnv,BioSeqPendEnv
from src.gfn.losses import (
    EdgeDecomposableLoss,
    Loss,
    Parametrization,
    StateDecomposableLoss,
    DBParametrization,
    TBParametrization,
    SubTBParametrization,
    TrajectoryDecomposableLoss,
    RLParametrization
)
from src.gfn.distributions import Empirical_Dist

def trajectories_to_training_samples(
    trajectories: Trajectories, loss_fn: Loss
) -> tuple[States,States] | Transitions | Trajectories:
    """Converts a Trajectories container to a States, Transitions or Trajectories container,
    depending on the loss.
    """
    if isinstance(loss_fn, StateDecomposableLoss):
        #return trajectories.to_states()
        return trajectories.intermediary_states,trajectories.last_states
    elif isinstance(loss_fn, TrajectoryDecomposableLoss):
        return trajectories
    elif isinstance(loss_fn, EdgeDecomposableLoss):
        return trajectories.to_transitions()
    else:
        raise ValueError(f"Loss {loss_fn} is not supported.")

def JSD(P, Q):
    """Computes the Jensen-Shannon divergence between two distributions P and Q"""
    P=torch.maximum(P,torch.tensor(1e-20))
    Q=torch.maximum(Q,torch.tensor(1e-20))
    M = 0.5 * (P + Q)
    return 0.5 * (torch.sum(P * torch.log(P / M)) + torch.sum(Q * torch.log(Q / M)))

def get_exact_P_T_Hypergrid(env,sampler):
    """This function evaluates the exact terminating state distribution P_T for HyperGrid.
    P_T(s') = u(s') P_F(s_f | s') where u(s') = \sum_{s \in Par(s')}  u(s) P_F(s' | s), and u(s_0) = 1
    """
    grid = env.build_grid()
    probabilities = sampler.actions_sampler.get_probs(grid)
    u = torch.ones(grid.batch_shape)
    all_states =env.all_states.states_tensor.tolist()
    for grid_ix in all_states[1:]:
        index = tuple(grid_ix)
        parents = [ index[:i] + (index[i] - 1,) + index[i + 1 :] + (i,)
                    for i in range(len(index)) if index[i] > 0] # parent, actions
        parents = torch.tensor(parents).T.numpy().tolist()
        u[index] = torch.sum(u[parents[:-1]] * probabilities[parents])
    return (u * probabilities[..., -1]).view(-1).detach().cpu()
def get_exact_P_T_bitseq(env,sampler):
    """
    This function evaluates the exact terminating state distribution P_T for graded DAG.
    :math:`P_T(s') = u(s') P_F(s_f | s')` where :math:`u(s') = \sum_{s \in Par(s')}  u(s) P_F(s' | s), and u(s_0) = 1`
    """
    ordered_states_list = env.ordered_states_list
    probabilities = sampler.actions_sampler.get_probs(env.all_states)
    u = torch.ones(size=env.all_states.batch_shape)
    for i, states in enumerate(ordered_states_list[1:]):
        #print(i + 1)
        index = env.get_states_indices(states)
        parents = torch.repeat_interleave(states.states_tensor,i+1,dim=0)

        backward_idx = torch.where(states.states_tensor != -1)[1]
        actions_idx =   env.ndim*parents[torch.arange(len(backward_idx)),backward_idx]+backward_idx
        parents[torch.arange(len(backward_idx)),backward_idx]=-1

        parents_idx = env.get_states_indices(env.States(parents))
        u[index] = (u[parents_idx] * probabilities[parents_idx, actions_idx]).reshape(-1, i + 1).sum(-1)

    return u[index].view(-1).cpu()

def get_exact_P_T_bitpend(env,sampler):
    """
    This function evaluates the exact terminating state distribution P_T for graded DAG.
    :math:`P_T(s') = u(s') P_F(s_f | s')` where :math:`u(s') = \sum_{s \in Par(s')}  u(s) P_F(s' | s), and u(s_0) = 1`
    """
    ordered_states_list = env.ordered_states_list
    probabilities =torch.zeros(size=(env.n_states,env.action_space.n))
    probabilities[env.get_states_indices(env.all_states)]=sampler.actions_sampler.get_probs(env.all_states)
    u = torch.ones(size=(probabilities.shape[0],))
    for i, states in enumerate(ordered_states_list[1:]):
        #print(i + 1)
        index = env.get_states_indices(states)
        parents = torch.repeat_interleave(states.states_tensor,2,dim=0)

        append_idx= (env.forward_index(states.states_tensor)-1).tolist()
        actions_idx  =  torch.stack([states.states_tensor[torch.arange(states.batch_shape[0]), append_idx]]
          + [states.states_tensor[:, 0] + env.nbase],dim=-1).flatten()

        odds_idx=torch.arange(1,states.batch_shape[0]*2,2)
        parents[ odds_idx,0:-1]= parents[ odds_idx,1:]
        parents[ odds_idx,-1]  =-1        #de-prepend
        parents[ odds_idx-1,append_idx] =-1 #de-append

        parents_idx = env.get_states_indices(env.States(parents))
        u[index] = (u[parents_idx] * probabilities[parents_idx, actions_idx]).reshape(-1, 2).sum(-1)

    return u[index].view(-1).cpu()

def get_exact_P_T(env, sampler):
    """This function evaluates the exact terminating state distribution P_T for DAG .
    P_T(s') = u(s') P_F(s_f | s') where u(s') = \sum_{s \in Par(s')}  u(s) P_F(s' | s), and u(s_0) = 1
    """
    all_states = env.all_states
    probabilities =  sampler.actions_sampler.get_probs(all_states)
    u = torch.ones(size=all_states.batch_shape)
    for i,state in enumerate(all_states[1:]):
        #print(i+1)
        parents = env.all_step(state[None,:], Backward=True)[state[None,:].backward_masks]
        parents_idx= env.get_states_indices(parents)
        actions_idx= torch.where(state[None,:].backward_masks)[1].tolist()
        u[i+1] = torch.sum(u[parents_idx] * probabilities[parents_idx,actions_idx])
    return (u * probabilities[..., -1]).view(-1).cpu()

def get_exact_P_T_G(env, sampler):
    """
    This function evaluates the exact terminating state distribution P_T for graded DAG.
    :math:`P_T(s') = u(s') P_F(s_f | s')` where :math:`u(s') = \sum_{s \in Par(s')}  u(s) P_F(s' | s), and u(s_0) = 1`
    """
    ordered_states_list = env.ordered_states_list
    probabilities =torch.zeros(size=(env.n_states,env.action_space.n))
    probabilities[env.get_states_indices(env.all_states)]=sampler.actions_sampler.get_probs(env.all_states)
    u=torch.ones(size=env.all_states.batch_shape)

    for i,states in enumerate(ordered_states_list[1:]):
        #print(i+1)
        num_parent_state=states.backward_masks.sum(-1)[0].item()# =i+1 for bitseq =2 for bitppend
        index   =  env.get_states_indices(states)
        parents = env.all_step(states, Backward=True)[states.backward_masks]
        actions_idx= env.bction2action( env.States(torch.repeat_interleave(states.states_tensor,2,dim=0)),
                                        torch.where(states.backward_masks)[1])#.tolist()
        parents_idx= env.get_states_indices(parents)
        u[index] = (u[parents_idx] * probabilities[parents_idx,actions_idx]).reshape(-1, num_parent_state).sum(-1)

    return u[index].view(-1).cpu()

def validate(
    env: Env,
    parametrization: Parametrization,
    sampler:TrajectoriesSampler,
    n_validation_samples: int = 1000,
    exact=False
) -> Dict[str, float]:
    """Evaluates the current parametrization on the given environment.
    This is for environments with known target reward. The validation is done by computing the l1 distance between the
    learned empirical and the target distributions.

    Args:
        env: The environment to evaluate the parametrization on.
        parametrization: The parametrization to evaluate.
        n_validation_samples: The number of samples to use to evaluate the pmf.

    Returns:
        Dict[str, float]: A dictionary containing the l1 validation metric. If the parametrization is a TBParametrization,
        i.e. contains LogZ, then the (absolute) difference between the learned and the target LogZ is also returned in the
        dictionary.
    """
    validation_info= {}
    if not exact:
        trajectories = sampler.sample_T(n_trajectories=n_validation_samples)
        final_states_dist= Empirical_Dist(env)
        final_states_dist_pmf = final_states_dist.pmf(trajectories)
        if isinstance(env,BitSeqEnv) or isinstance(env,BioSeqPendEnv):
            validation_info["mean_diff"] = (env.log_reward(trajectories).exp().mean() / env.mean_reward).clamp(0,1).item()
    else:
        true_dist_pmf = env.true_dist_pmf
        if isinstance(env, HyperGrid):
            final_states_dist_pmf = get_exact_P_T_Hypergrid(env, sampler)
        elif  isinstance(env,DAG_BN):
            final_states_dist_pmf = get_exact_P_T(env, sampler)
        elif isinstance(env,BitSeqEnv):
            final_states_dist_pmf = get_exact_P_T_bitseq(env, sampler)
        elif isinstance(env,BioSeqPendEnv):
            final_states_dist_pmf = get_exact_P_T_bitpend(env, sampler)
        else:
            raise ValueError("Environment Not suppoerted")
        if  isinstance(env,BitSeqEnv) or isinstance(env,BioSeqPendEnv):
            est_reward = (final_states_dist_pmf* env.log_reward(env.terminating_states).exp()).sum()
            validation_info["mean_diff"] =  (est_reward / env.mean_reward).clamp(0,1).item()
        validation_info["l1_dist"]= 0.5*torch.abs(final_states_dist_pmf - true_dist_pmf).sum().item()
        validation_info["JSD"]= JSD(final_states_dist_pmf,true_dist_pmf).item()

    true_logZ = env.log_partition
    if isinstance(parametrization, TBParametrization) | isinstance(parametrization, RLParametrization):
        logZ = parametrization.logZ.tensor
    elif isinstance(parametrization,DBParametrization)|isinstance(parametrization,SubTBParametrization):
        logZ= parametrization.logF(env.States(env.s0))
    validation_info["Z_diff"] = abs((logZ.exp() - true_logZ.exp()).item())
    validation_info["logZ_diff"] = abs((logZ - true_logZ).item())

    if hasattr(env, 'replay_x'):
        trajectories = sampler.sample_T(n_trajectories=len(env.oracle.modes))#tf8 256 qm 768 tf10 5000
        env.replay_x.add( trajectories, env.log_reward(trajectories).exp())
        #validation_info["mean_diff"] = (env.replay_x.terminating_rewards[-50000:].mean() / env.mean_reward).clamp(0,1).item()
        validation_info["num_modes"]= env.oracle.is_index_modes[torch.unique(env.replay_x.x_index)].sum().item()
    return validation_info, final_states_dist_pmf

import networkx as nx
from itertools import permutations, product,chain,combinations
def check_acylic(states_tensor):
    is_directed = []
    for edges in states_tensor:
        edges = edges.reshape( int(edges.shape[-1]**0.5),
                               int(edges.shape[-1]**0.5)).numpy()
        G = nx.DiGraph(edges)
        is_directed.append(nx.is_directed_acyclic_graph(G))
    return all(is_directed)

def all_dag(n_nodes,n_edges=25):
    nodelist = list(range(n_nodes))
    edges = list(permutations(nodelist, 2))  # n*(n-1) possible directed edges
    all_graphs = chain.from_iterable(combinations(edges, r) for r in range(len(edges) + 1)) #power set

    for graph_edges in all_graphs:
        if len(graph_edges)>n_edges:
            continue
        graph = nx.DiGraph(graph_edges)
        graph.add_nodes_from(nodelist)
        if nx.is_directed_acyclic_graph(graph):
            str_adj= nx.to_numpy_array(graph,dtype=int,nodelist=sorted(graph.nodes)).flatten()
            yield  str_adj

##################################
# for param operation
##################################
def set_flat_params_to(model, flat_params):
    prev_ind = 0
    for param in model.parameters():
        flat_size = param.numel()
        param.data.copy_(flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
        prev_ind += flat_size