from src.gfn.estimators import (
    LogEdgeFlowEstimator,
    LogitPBEstimator,
    LogitPFEstimator,
    LogStateFlowEstimator,
    LogZEstimator,
)
from src.gfn.losses import (
    Loss,
    DetailedBalance,
    FlowMatching,
    TrajectoryBalance,
    SubTrajectoryBalance,
    TrajectoryRL,
    Trajectory_TRPO
)
from src.gfn.losses import (
    Parametrization,
    DBParametrization,
    FMParametrization,
    PFBasedParametrization,
    SubTBParametrization,
    TBParametrization,
    RLParametrization,
)
from src.gfn.envs import Env,BitSeqEnv,HyperGrid,DAG_BN,BioSeqEnv,BioSeqPendEnv
from src.gfn.samplers import DiscreteActionsSampler, TrajectoriesSampler,BackwardDiscreteActionsSampler
from data_dag.factories import get_scorer
from src.gfn.utils import all_dag
import networkx as nx
import torch
from typing import Tuple
import numpy as np
def EnvConfig(args):
    if args.Env=='HyperGrid':
        env=HyperGrid(ndim=args.ndim,
                      height=args.height,
                      R0=args.R0,R1=0.5,R2=2.0,
                      reward_cos=False,
                      preprocessor_name="KHot")
    elif args.Env=="BayesianNetwork":
        scorer, data, graph = get_scorer(args)
        #true_graph = torch.tensor(nx.to_numpy_array(graph, nodelist=sorted(graph.nodes), weight=None))
        all_graphs = torch.tensor(np.load('DAG-5-list.npy')) #all_graphs=list(all_dag(5))
        env=DAG_BN(n_dim=5,all_graphs=all_graphs,score=scorer)
    elif args.Env == "TFbind8":
        env = BioSeqEnv(ndim=8, nbase=4,
                        oracle_path='data_bio/tfbind8/tfbind8-exact-v0-all.pkl',
                        mode_path='data_bio/tfbind8/modes_tfbind8.pkl',alpha=3, R_max=10, R_min=1e-3,name="TFbind8")
    elif args.Env == "qm9str":
        env = BioSeqEnv(ndim=5, nbase=11,
                        oracle_path='data_bio/qm9str/block_qm9str_v1_s5.pkl',alpha=5,R_max=10,R_min=1e-3,name="qm9str")
    elif args.Env=="sehstr":
        env = BioSeqEnv(ndim=6, nbase=18,
                        oracle_path='data_bio/sehstr/block_18_stop6.pkl', alpha=6, R_max=10, R_min=1e-3,name="sehstr")
    elif args.Env == "TFbind10":
        env = BioSeqEnv(ndim=10, nbase=4,
                        oracle_path='data_bio/tfbind10/tfbind10-exact-v0-all.pkl', alpha=3, R_max=10, R_min=0,name="TFbind10")
    else:
        raise "no environment supported"
    return env

def SamplerConfig(
        env: Env,
        parametrization: Parametrization) -> [TrajectoriesSampler,TrajectoriesSampler]:
    if isinstance(parametrization, FMParametrization):
        estimator,B_estimator  = parametrization.logF,None
    elif isinstance(parametrization, PFBasedParametrization):
        estimator,B_estimator  = parametrization.logit_PF,parametrization.logit_PB
    else:
        raise ValueError(f"Cannot parse sampler for parametrization {parametrization}")
    actions_sampler = DiscreteActionsSampler(estimator=estimator)
    B_actions_sampler=BackwardDiscreteActionsSampler(estimator=B_estimator)
    trajectories_sampler = TrajectoriesSampler(env=env, actions_sampler=actions_sampler)
    B_trajectories_sampler = TrajectoriesSampler(env=env, actions_sampler=B_actions_sampler)
    return trajectories_sampler,B_trajectories_sampler

def OptimConfig(parametrization: Parametrization,lr=0.001,lr_Z=0.1,lr_V=0.005,logV=None,logVB=None):
    if  not isinstance(parametrization, RLParametrization):
        params = [{"params":param ,"lr": lr if estimator != "logZ" else lr_Z}
                  for estimator,param in parametrization.parameters.items()]
        optimizer = torch.optim.Adam(params)
        return optimizer
    else:
        params = [{"params": parametrization.logit_PF.parameters(), "lr": lr}]\
                 +[{"params": parametrization.logZ.parameters(), "lr": lr_Z}]
        optimizer = torch.optim.Adam(params)
        B_optimizer = torch.optim.Adam(parametrization.logit_PB.parameters(),lr) if len(parametrization.logit_PB.parameters()) else None
        V_optimizer = torch.optim.Adam(logV.parameters(), lr_V)
        # V_optimizer = torch.optim.LBFGS(logV.parameters(),history_size=10, max_iter=4)
        VB_optimizer = torch.optim.Adam(logVB.parameters(), lr_V)
        return optimizer, V_optimizer,B_optimizer, VB_optimizer

def get_estimators(env:Env,
                   PB_parameterized,
                   **GFNModuleConfig)-> Tuple[LogitPFEstimator, LogitPBEstimator,
          LogStateFlowEstimator,LogEdgeFlowEstimator]:

    logit_PF=logit_PB =logF_state=logF_edge = GFNModuleConfig
    logit_PF = LogitPFEstimator(env=env, **logit_PF)

    if PB_parameterized:
        logit_PB = LogitPBEstimator(env=env, **logit_PB)
    else:
        logit_PB = LogitPBEstimator(env=env,module_name= 'Uniform')
    logF_state = LogStateFlowEstimator(env=env, **logF_state)
    logF_edge  =  LogEdgeFlowEstimator(env=env, **logF_edge)
    return (logit_PF, logit_PB, logF_state,logF_edge)

def FMLossConfig(env:Env,args):
    _,_,_,logF_edge = get_estimators(env=env, **args.GFNModuleConfig)
    parametrization = FMParametrization(logF_edge)
    optimizer=OptimConfig(parametrization,**args.optim)
    loss = FlowMatching(parametrization,optimizer)
    return parametrization, loss

def DBLossConfig(env:Env,args):
    logit_PF,logit_PB,logF_state,_ = get_estimators(env=env,PB_parameterized=args.PB_parameterized,**args.GFNModuleConfig)
    parametrization = DBParametrization(logit_PF, logit_PB, logF_state)
    optimizer = OptimConfig(parametrization, **args.optim)
    loss = DetailedBalance(parametrization,optimizer)
    return (parametrization, loss)

def SubTBLossConfig(env:Env,args,log_reward_clip_min: float = -12):
    logit_PF, logit_PB, logF_state,_ = get_estimators(env=env,PB_parameterized=args.PB_parameterized,**args.GFNModuleConfig )
    parametrization = SubTBParametrization(logit_PF, logit_PB, logF_state)
    optimizer = OptimConfig(parametrization, **args.optim)
    loss = SubTrajectoryBalance(parametrization,optimizer,log_reward_clip_min=log_reward_clip_min,lamda=args.lamb)
    return (parametrization, loss)

def TBLossConfig(env:Env,args,logZ_init: float = 0.0,log_reward_clip_min: float = -12):
    logit_PF, logit_PB,_,_= get_estimators(env=env,PB_parameterized=args.PB_parameterized,**args.GFNModuleConfig)
    logZ = LogZEstimator(tensor=torch.tensor( logZ_init, dtype=torch.float))
    parametrization = TBParametrization(logit_PF, logit_PB, logZ)
    optimizer = OptimConfig(parametrization, **args.optim)
    loss = TrajectoryBalance(parametrization,optimizer,log_reward_clip_min=log_reward_clip_min)
    return (parametrization, loss)

def RLLossConfig(env:Env,args,logZ_init: float = 0.0,log_reward_clip_min: float = -12):
    logit_PF, logit_PB,_,_= get_estimators(env=env,PB_parameterized=args.PB_parameterized,**args.GFNModuleConfig)
    logZ = LogZEstimator(tensor=torch.tensor(logZ_init, dtype=torch.float))
    logV= LogStateFlowEstimator(env=env, module_name="NeuralNet")
    logVB= LogStateFlowEstimator(env=env, module_name="NeuralNet")
    parametrization = RLParametrization(logit_PF, logit_PB, logZ)
    optimizer = OptimConfig(parametrization,logV=logV,logVB=logVB, **args.optim)
    loss = TrajectoryRL(parametrization,optimizer,logV,logVB, PG_used=args.PG_used,lamb=args.lamb,
                        log_reward_clip_min=log_reward_clip_min,env=env)
    return (parametrization, loss)


def TRPOLossConfig(env:Env,args,logZ_init: float = 0.0,log_reward_clip_min: float = -12):
    logit_PF, logit_PB,_,_= get_estimators(env=env,PB_parameterized=args.PB_parameterized,**args.GFNModuleConfig)
    logZ = LogZEstimator(tensor=torch.tensor(logZ_init, dtype=torch.float))
    logV= LogStateFlowEstimator(env=env, module_name="NeuralNet")
    logVB= LogStateFlowEstimator(env=env, module_name="NeuralNet")
    parametrization = RLParametrization(logit_PF, logit_PB, logZ)
    optimizer = OptimConfig(parametrization,logV=logV,logVB=logVB, **args.optim)
    loss = Trajectory_TRPO(parametrization,optimizer,logV,logVB,PG_used=args.PG_used,lamb=args.lamb,
                        log_reward_clip_min=log_reward_clip_min,env=env)
    return (parametrization, loss)

def Config(args):
    env = EnvConfig(args)
    if args.Loss=='FM':
        parametrization, loss=FMLossConfig(env,args)
    elif args.Loss=="DB":
        parametrization, loss=DBLossConfig(env,args)
    elif args.Loss == "TB":
        parametrization, loss = TBLossConfig(env,args)
    elif args.Loss == "Sub_TB":
        parametrization, loss= SubTBLossConfig(env,args)
    elif args.Loss == "RL":
        parametrization, loss= RLLossConfig(env,args)
    elif args.Loss == "TRPO":
        parametrization, loss= TRPOLossConfig(env,args)
    else:
        raise 'loss function not implemented'
    return env,parametrization,loss
