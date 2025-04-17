from typing import List, Literal, Tuple

import torch
from torchtyping import TensorType
from src.gfn.containers import Trajectories
from src.gfn.envs   import Env
from src.gfn.estimators import LogStateFlowEstimator,LogZEstimator,LogitPBEstimator
from src.gfn.losses.base import PFBasedParametrization, TrajectoryDecomposableLoss
from src.gfn.samplers import BackwardDiscreteActionsSampler
from src.gfn.containers.states import States
# Typing
ScoresTensor = TensorType[-1, float]
LossTensor = TensorType[0, float]
LogPTrajectoriesTensor = TensorType["max_length", "n_trajectories", float]


class RLParametrization(PFBasedParametrization):
    r"""
    Exactly the same as DBParametrization
    """
    def __init__(self, logit_PF,logit_PB, logZ: LogZEstimator):
        self.logZ=logZ
        super().__init__(logit_PF,logit_PB)

class TrajectoryRL(TrajectoryDecomposableLoss):
    def __init__(
        self,
        parametrization: RLParametrization,
        optimizer: Tuple[torch.optim.Optimizer,
                         torch.optim.Optimizer,
                         torch.optim.Optimizer | None,
                         torch.optim.Optimizer | None],
        logV: LogStateFlowEstimator,
        logVB: LogStateFlowEstimator,
        env: Env,
        PG_used:bool=False,
        log_reward_clip_min: float = -12,
        lamb:float=1.0,
    ):
        """
        Args:
             lamda: parameter for bias-variance trade-off
        """
        self.lamb  = lamb
        self.logV=logV
        self.logVB=logVB
        self.PG_used=PG_used
        self.env=env
        self.log_reward_clip_min=log_reward_clip_min
        self.A_optimizer,self.V_optimizer,\
        self.B_optimizer,self.VB_optimizer=optimizer
        super().__init__(parametrization,fill_value=0.)

    def DAG_BN(self,trajectories:Trajectories):
        valid_states, valid_actions, valid_index = self.forward_state_actions(trajectories)
        with torch.no_grad():
            logits = self.parametrization.logit_PF(valid_states)
        if torch.any(torch.all(torch.isnan(logits), 1)): raise ValueError("NaNs in estimator")
        logits[~valid_states.forward_masks] = self.inf_value
        low_states_index = ~(self.env.log_reward(valid_states) > torch.tensor(75.))
        logits[low_states_index, -1] = logits[low_states_index,-1].min(torch.tensor(1e-10).log())#self.inf_value
        #
        log_pg_all = logits.log_softmax(dim=-1)
        log_pg_actions = self.action_prob_gather(log_pg_all, valid_actions)
        #
        log_pg_trajectories = torch.full_like(trajectories.actions, fill_value=self.fill_value, dtype=torch.float)
        log_pg_trajectories[valid_index] = log_pg_actions
        return  log_pg_trajectories


    def HyperGrid(self,trajectories:Trajectories): # for hyper-grid
        valid_states,valid_actions,valid_index=self.forward_state_actions(trajectories)
        with torch.no_grad():
            logits=self.parametrization.logit_PF(valid_states)
        if torch.any(torch.all(torch.isnan(logits), 1)): raise ValueError("NaNs in estimator")
        logits[~valid_states.forward_masks] = self.inf_value
        low_states_index = ~(self.env.log_reward(valid_states) > torch.tensor([self.env.R0]).log())
        logits[low_states_index, -1] = logits[low_states_index, -1].min(torch.tensor(1e-5).log())#self.inf_value
        #
        log_pg_all = logits.log_softmax(dim=-1)
        log_pg_actions = self.action_prob_gather(log_pg_all, valid_actions)
        #
        log_pg_trajectories = torch.full_like(trajectories.actions, fill_value=self.fill_value, dtype=torch.float)
        log_pg_trajectories[valid_index] = log_pg_actions
        return  log_pg_trajectories

    def BitSeqEnv(self,trajectories:Trajectories):
        return NotImplementedError("The environment does not support guided distribution")

    def BioSeqEnv_approx(self,trajectories:Trajectories):
        if hasattr(self.env,'replay_G'):
            backward=trajectories.is_backward
            self.env.replay_G.add(trajectories.states[0] if backward else trajectories.last_states,
                                  trajectories.log_rewards.exp() )
            log_pg_traj=trajectories.log_probs
            valid_traj=trajectories.states[1:] if  backward else trajectories.states[:-1] # backward s_{T-1}....s_0,    forward s_0->...s_{T-1}->x
            valid_trajn=trajectories.states[:-1] if  backward else trajectories.states[1:] # backward  x->s_{T-1}....s_1, forward s_1->... x->sf
            orders=list(reversed(range(valid_traj.batch_shape[0]))) if  backward \
                else list(range(valid_traj.batch_shape[0]))
            for batch in range(valid_traj.batch_shape[1]):
                valid_order_index = None
                for order in orders:
                    if trajectories.actions[order,batch]!=-1:
                        state= valid_traj[order,batch]
                        nstate= valid_trajn[order,batch]
                        scores,scores_Z,valid_order_index = self.env.replay_G.scores_approx(state,nstate,valid_order_index)
                        if scores > 0:
                            log_pg_traj[order,batch]=(scores/scores_Z).log().maximum(torch.tensor(self.inf_value))
        else:
            log_pg_traj=trajectories.log_probs
        return log_pg_traj

    def BioSeqEnv(self,trajectories:Trajectories):
        if hasattr(self.env,'replay_G'):
            backward=trajectories.is_backward
            last_states=trajectories.states[0] if backward else trajectories.last_states
            self.env.replay_G.add( last_states,trajectories.log_rewards.exp() )
            log_pg_traj=trajectories.log_probs
            valid_traj=trajectories.states[1:] if  backward else trajectories.states[:-1] # backward s_{T-1}....s_0,    forward s_0->...s_{T-1}->x
            orders=list(reversed(range(valid_traj.batch_shape[0]))) if  backward else list(range(valid_traj.batch_shape[0]))
            for batch in range(valid_traj.batch_shape[1]):
                valid_order_index = None
                tstate =   last_states[batch]
                actions_list=torch.full_like(tstate.forward_masks,fill_value=False,dtype=torch.bool)
                actions_list[tstate.states_tensor * trajectories.env.ndim+ torch.arange(0, trajectories.env.ndim)]=True
                for order in orders:
                    if trajectories.actions[order,batch]!=-1:
                        actions= trajectories.actions[order,batch]
                        state  = valid_traj[order, batch]
                        nstate = trajectories.env.all_step(state)
                        nstate_list=actions_list & state.forward_masks
                        scores,valid_order_index = self.env.replay_G.scores(state,nstate,nstate_list,valid_order_index)
                        if scores.sum() > 0:
                            log_pg_traj[order,batch]=(scores[actions]/scores.sum()).log().maximum(torch.tensor(self.inf_value))
        else:
            log_pg_traj=trajectories.log_probs
        return log_pg_traj

    def get_pgs(self,trajectories: Trajectories) -> LogPTrajectoriesTensor:
        return self.__getattribute__(self.env.__class__.__name__)(trajectories)

    def get_scores(
        self, trajectories: Trajectories,
            log_pf_traj:LogPTrajectoriesTensor,
            log_pb_traj:LogPTrajectoriesTensor,
    ) -> ScoresTensor:

        terminal_index=trajectories.is_terminating_action
        log_pb_traj.T[terminal_index.T]=trajectories.log_rewards.clamp_min(self.log_reward_clip_min)\
                                                -self.parametrization.logZ.tensor
        scores= (log_pf_traj-log_pb_traj)
        return  scores

    def get_value(self,trajectories:Trajectories,backward=False):
        flatten_masks = ~trajectories.is_sink_action
        values = torch.full_like(trajectories.actions,dtype=torch.float, fill_value=self.fill_value)
        valid_states = trajectories.states[:-1][flatten_masks]  # remove the dummpy one extra sink states
        values[flatten_masks] = self.logV(valid_states).squeeze(-1) if not backward \
            else self.logVB(valid_states).squeeze(-1)
        return values

    def surrogate_loss(self,log_pf, log_qf,advantages):
        """define the loss objective for TRPO"""
        # Its value:    adv
        # Its gradient: adv *▽log p  (= adv* (▽p/p)= ad * {▽exp(logp)/exp(logp)} )
        sur_loss=torch.exp(log_pf - log_qf.detach()).mul(advantages)
        return sur_loss

    def update_model(self, trajectories: Trajectories):
        log_pf_traj= self.get_pfs(trajectories)
        log_pb_traj= self.get_pbs(trajectories)

        scores = self.get_scores(trajectories, log_pf_traj, log_pb_traj).detach()
        values = self.get_value(trajectories)
        advantages, targets = self.estimate_advantages(trajectories, scores, values.detach())
        Z = self.parametrization.logZ.tensor.exp()
        A_loss = self.surrogate_loss(log_pf_traj, log_pf_traj, advantages).sum(0).mean()
        Z_diff = (Z / Z.detach()) * (scores.sum(0).mean())
        V_loss = ((targets - values).pow(2)).sum(0).mean()

        if isinstance(self.V_optimizer,torch.optim.LBFGS):
            def closure():
                self.V_optimizer.zero_grad()
                val=self.get_value(trajectories)
                V_loss= (targets-val).pow(2).sum(0).mean()
                V_loss.backward()
                return V_loss
            self.V_optimizer.step(closure)
        else:
            self.optimizer_step(V_loss, self.V_optimizer)
        self.optimizer_step(A_loss + Z_diff, self.A_optimizer)
        return A_loss + Z_diff

    def B_update_model(self, trajectories: Trajectories):
        log_pb_traj= self.get_pbs(trajectories)
        if self.PG_used:
            log_pg_traj = self.get_pgs(trajectories)
        else:
            log_pg_traj = self.get_pfs(trajectories)
        scores = (log_pb_traj - log_pg_traj).detach()
        values = self.get_value(trajectories, backward=True)
        advantages, targets = self.estimate_advantages(trajectories, scores, values.detach(),unbias=True)
        A_loss = self.surrogate_loss(log_pb_traj, log_pb_traj, advantages).sum(0).mean()
        V_loss = (targets - values).pow(2).sum(0).mean()
        # Kl=self.kl_log_prob(log_pb_traj_all,log_pg_traj_all).mean()

        self.optimizer_step(V_loss, self.VB_optimizer)
        self.optimizer_step(A_loss, self.B_optimizer)
        return A_loss  # ,Kl.detach()

    def B_update_model_Emp(self, B_trajectories: Trajectories):
        trajectories=B_trajectories.revert_backward_trajectories()
        log_pb_traj= self.get_pbs(trajectories)
        if self.PG_used:
            log_pg_traj = self.get_pgs(trajectories)
        else:
            log_pg_traj = self.get_pfs(trajectories)
        scores = self.get_scores(trajectories, log_pg_traj.detach(), log_pb_traj)
        loss=scores.sum(0).pow(2).mean()
        self.optimizer_step(loss, self.B_optimizer)
        return loss  # ,Kl.detach()

    def optimizer_step(self, loss, optimizer):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def __call__(self, trajectories: Trajectories) -> Tuple[LossTensor, LossTensor]:
        pass

    @staticmethod
    def kl_log_prob(log_prob_q, log_prob_p):
        log_prob_p = log_prob_p.detach()
        kl = (log_prob_p.exp() * (log_prob_p - log_prob_q)).sum(-1)
        return kl
    @staticmethod
    def entropy(log_pf):
        p_log_p = -(log_pf * log_pf.exp()).sum(-1)
        return p_log_p

    def estimate_advantages(self,trajectories:Trajectories,scores,values,unbias=False):
        """
        Returns:
            -tar: estimated :math:`\hat{V}t` for the optimzation of  :math:`V(\eta)` advantages
            -advantages:  estimated advantage function
        """
        lamb = 1. if unbias else self.lamb
        masks = ~trajectories.is_sink_action
        Vt_prev = torch.zeros_like(scores[0], dtype=torch.float)
        adv_prev = torch.zeros_like(scores[0], dtype=torch.float)
        deltas = torch.full_like(scores[0], fill_value=0., dtype=torch.float)
        tar_prev = torch.zeros_like(scores[0], dtype=torch.float)
        tar = torch.full_like(scores, fill_value=0., dtype=torch.float)
        advantages = torch.full_like(scores, fill_value=0., dtype=torch.float)
        for i in reversed(range(scores.size(0))):
            tar_prev[masks[i]] = scores[i][masks[i]] + tar_prev[
                masks[i]]  #T-step estimation  or + Vt_prev[masks[i]] one-step estimation
            tar[i][masks[i]] = tar_prev[masks[i]]
            #########################################
            deltas[masks[i]] = scores[i][masks[i]] +  Vt_prev[masks[i]] - values[i][masks[i]] #if lamb!=1 else scores[i][masks[i]]
            if torch.any(torch.isnan(deltas)): raise ValueError("NaN in scores")
            Vt_prev = values[i]
            adv_prev[masks[i]]= deltas[masks[i]] + lamb * adv_prev[masks[i]]
            advantages[i][masks[i]]= adv_prev[masks[i]]
        advantages[masks] = (advantages[masks] - advantages[masks].mean())
        return advantages, tar

