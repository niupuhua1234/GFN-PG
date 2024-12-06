import math
import pandas as pd
from abc import ABC, abstractmethod
import torch
class BaseScore(ABC):
    """Base class for the scorer.

    Parameters
    ----------
    data : pd.DataFrame
        The dataset.

    prior : `BasePrior` instance
        The prior over graphs p(G).
    """

    def __init__(self, data, prior):
        self.data = data
        self.prior = prior
        self.column_names = list(data.columns)
        self.num_variables = len(self.column_names)
        self.prior.num_variables = self.num_variables


    def __call__(self, graphs):
        assert graphs.ndim==1 or graphs.ndim ==2
        if graphs.ndim<2:
            graphs=graphs[None,:]
        batch_size=graphs.shape[0]
        graphs = graphs.reshape(batch_size,self.num_variables, self.num_variables).float()
        scores=torch.zeros(batch_size)
        for i, graph in enumerate(graphs):
            scores[i] += self.structure_prior(graph)
        with torch.no_grad():
            for node,_ in enumerate(self.column_names):
                edge_idx = torch.nonzero(graphs[:,:,node]) #batch_idx, edge_idx
                par_idx  = torch.arange(self.num_variables, 2 * self.num_variables).repeat(batch_size, 1)
                par_idx[edge_idx[:, 0], edge_idx[:, 1]] = edge_idx[:, 1]
                par_num=graphs[:,:,node].sum(-1).long()
                scores= scores + self.local_scores(node, par_idx,par_num)
        return scores
    def single_call(self,graphs):
        assert graphs.ndim == 1 or graphs.ndim == 2
        if graphs.ndim < 2:
            graphs = graphs[None, :]
        batch_size = graphs.shape[0]
        graphs = graphs.reshape(batch_size, self.num_variables, self.num_variables).float()
        scores = torch.zeros(batch_size,dtype=torch.float)
        for i, graph in enumerate(graphs):
            scores[i] += self.structure_prior(graph)
        for i,graph in enumerate(graphs):
            print(i)
            local_score=0.
            for node,_ in enumerate(self.column_names):
                edge_idx = torch.nonzero(graph[ :, node]).squeeze()
                par_num = graph[ :, node].sum(-1).long()
                local_score = local_score + self.local_scores_s(node, edge_idx, par_num)
            scores[i]=local_score
        return torch.tensor(scores)
    @abstractmethod
    def local_scores(self, target, indices,indices_num):
        pass
    @abstractmethod
    def local_scores_s(self, target, indices,indices_num):
        pass
    def structure_prior(self, graph):
        """A (log) prior distribution over models. Currently unused (= uniform)."""
        return 0

def logdet(array):
    _, logdet = torch.slogdet(array)
    return logdet
def ix_(array):
    return array[...,:,None],array[...,None,:]

class BGeScore(BaseScore):
    r"""BGe score.

    Parameters
    ----------
    data : pd.DataFrame
        A DataFrame containing the (continuous) dataset D. Each column
        corresponds to one variable. The dataset D is assumed to only
        contain observational data_bio (a `INT` column will be treated as
        a continuous variable like any other).

    prior : `BasePrior` instance
        The prior over graphs p(G).

    mean_obs : np.ndarray (optional)
        Mean parameter of the Normal prior over the mean $\mu$. This array must
        have size `(N,)`, where `N` is the number of variables. By default,
        the mean parameter is 0.

    alpha_mu : float (default: 1.)
        Parameter $\alpha_{\mu}$ corresponding to the precision parameter
        of the Normal prior over the mean $\mu$.

    alpha_w : float (optional)
        Parameter $\alpha_{w}$ corresponding to the number of degrees of
        freedom of the Wishart prior of the precision matrix $W$. This
        parameter must satisfy `alpha_w > N - 1`, where `N` is the number
        of varaibles. By default, `alpha_w = N + 2`.
    """
    def __init__(self,data,prior,mean_obs=None,alpha_mu=1.,alpha_w=None):
        num_variables = data.shape[1]
        if mean_obs is None:
            mean_obs = torch.zeros((num_variables,),dtype=torch.float)
        if alpha_w is None:
            alpha_w = num_variables + 2.

        super().__init__(data, prior)
        self.mean_obs = mean_obs
        self.alpha_mu = alpha_mu
        self.alpha_w = alpha_w

        self.num_samples = self.data.shape[0]
        self.t = (self.alpha_mu * (self.alpha_w - self.num_variables - 1)) / (self.alpha_mu + 1)
        #self.t=self.alpha_mu/(self.alpha_mu+self.num_samples)

        T = self.t * torch.eye(self.num_variables)
        #T = torch.eye(self.num_variables)# assuem W^-1 of wishart prior is I
        data = torch.tensor(self.data.values,dtype=torch.float)
        data_mean = torch.mean(data, dim=0, keepdim=True)
        data_centered = data - data_mean

        self.R = (T + torch.matmul(data_centered.T, data_centered)  #                        T+S_N
                  + ((self.num_samples * self.alpha_mu) / (self.num_samples + self.alpha_mu))
                  * torch.matmul((data_mean - self.mean_obs).T, data_mean - self.mean_obs)   # (N*α_μ)/(N+α_μ)*(v-x_μ)(v-x_μ)T
                  )
        self.block_R_I = torch.block_diag(self.R, torch.eye(self.num_variables))
        all_parents = torch.arange(self.num_variables)
        self.log_gamma_term = (
            0.5 * (math.log(self.alpha_mu) - math.log(self.num_samples + self.alpha_mu))
            + torch.lgamma(0.5 * (self.num_samples + self.alpha_w - self.num_variables + all_parents + 1))  #log Γ((N+α_w-n+l)/2)  for l=0,...n      log  Γ_l+1((N+α_w-n+l+1)/2) - log Γ_l((N+α_w-n)/2) =log  Γ((N+α_w-n+l)/2)
            - torch.lgamma(0.5 * (self.alpha_w - self.num_variables + all_parents + 1))                     #-log Γ((α_w-n+l)/2)  for l=0,...n       -( log Γ_l+1((α_w-n+l+1)/2)- log Γ_l((α_w-n)/2))=   log
            - 0.5 * self.num_samples * math.log(math.pi)                                                    #π^(l*N- (l+1)N) =π^N
            + 0.5 * (self.alpha_w - self.num_variables + 2 * all_parents + 1) * math.log(self.t)            # +0.5*math.log(self.t)
        )
        # batch-wsie compute score sum
        # idea compute the score of [R   0]
        #                           [0,  I]
        # parents = torch.arange(self.num_variables, 2 * self.num_variables).repeat(batch_size, 1)
        # edge_idx =graphs[:,:,node].nonzero()
        # parents[edge_idx[:, 0], edge_idx[:, 1]] = edge_idx[:, 1]
        #
        # block_R_I = torch.block_diag(torch.tensor(self.R), torch.eye(self.num_variables))
    def local_scores(self,target,indices,indices_num):
        num_parents = indices_num.clone()
        variables = torch.clone(indices)
        variables[torch.arange(len(indices_num)),target]=target
        log_term_r = (0.5 * (self.num_samples + self.alpha_w - self.num_variables + num_parents)
                      * logdet(self.block_R_I[ix_(indices)])
                      - 0.5 * (self.num_samples + self.alpha_w - self.num_variables + num_parents + 1)
                      * logdet(self.block_R_I[ix_(variables)])
                      )
        return self.log_gamma_term[num_parents] + log_term_r

    def local_scores_s(self,target,indices,indices_num):
        num_parents = indices_num.clone()
        indices = indices.view(1) if indices.ndim == 0 else indices
        if num_parents >0:
            variables = torch.cat((torch.tensor(target).unsqueeze(0),indices))
            log_term_r = (
                0.5 * (self.num_samples + self.alpha_w - self.num_variables + num_parents)   # log{ |R_YY|^(N+a_w-n+l)/2}
                * logdet(self.R[ix_(indices)])
                - 0.5 * (self.num_samples + self.alpha_w - self.num_variables + num_parents + 1)
                * logdet(self.R[ix_(variables)])
            )
        else:
            log_term_r = (-0.5 * (self.num_samples + self.alpha_w - self.num_variables + 1)
                * torch.log(torch.abs(self.R[target, target])))
        return self.log_gamma_term[num_parents]  + log_term_r