from numpy.random import default_rng

from data_dag.scores import  BGeScore,priors
from data_dag.data  import get_data


def get_prior(name, **kwargs):
    prior = {
        'uniform': priors.UniformPrior,
        'erdos_renyi': priors.ErdosRenyiPrior,
        'edge': priors.EdgePrior,
        'fair': priors.FairPrior
    }
    return prior[name](**kwargs)

def get_scorer(args,rng=default_rng(0)):
    # Get the data_bio
    graph, data, score = get_data(args.graph, args, rng=rng)
    data = data[list(graph.nodes)]
    # Get the prior
    prior = get_prior(args.prior, **args.prior_kwargs)
    scorer = BGeScore(data, prior, **args.scorer_kwargs)

    return scorer, data, graph
