import torch
import wandb
from datetime import  datetime
from Config import Config,SamplerConfig
from src.gfn.containers.replay_buffer import ReplayBuffer
from src.gfn.utils import trajectories_to_training_samples, validate


from argparse import ArgumentParser,BooleanOptionalAction
from simple_parsing.helpers.serialization import encode
from tqdm import tqdm, trange
timestamp = datetime.now().strftime('-%Y%m%d_%H%M%S')
parser = ArgumentParser(description='DAG-GFlowNet-new')
parser.add_argument('--project',default='DAG_BN_Eval')
# Environment
environment = parser.add_argument_group('Environment')
environment.add_argument('--Env',default='BayesianNetwork', choices=['BayesianNetwork'])
bn = parser.add_argument_group('Type of graph')
bn.add_argument('--graph', default='erdos_renyi_lingauss', choices=['erdos_renyi_lingauss', 'sachs_continuous'])
bn.add_argument('--prior', type=str, default='uniform', choices=['uniform', 'erdos_renyi', 'edge', 'fair'])
bn.add_argument('--prior_kwargs', type=dict, default={}, help='Arguments of the prior over graphs.')
bn.add_argument('--scorer_kwargs', type=dict, default={},help='Arguments of the scorer.')
bn.add_argument('--num_samples', type=int, default=100)
bn.add_argument('--num_variables', type=int, default=5, help='design the number of nodes in erdos graph')
bn.add_argument('--num_edges', type=int, default=5, help='design the number of edges in erdos graph')
#Model
optimization = parser.add_argument_group('Method')
optimization.add_argument("--PB_parameterized",default=False)
optimization.add_argument("--PG_used",default=False)
optimization.add_argument("--lamb",type=float,default=0.99)
optimization.add_argument("--epsilon_decay",type=float,default=0.99)
optimization.add_argument("--epsilon_start",type=float,default=0.0)
optimization.add_argument("--epsilon_end",type=float,default=0.0)
# Optimization
optimization = parser.add_argument_group('Optimization')
optimization.add_argument('--Loss',default='RL', choices=['DB','TB','RL','TRPO'])
optimization.add_argument("--seed", type=int, default=2)
optimization.add_argument("--optim",default={'lr':0.001,'lr_Z':1.0  ,'lr_V':0.005})
optimization.add_argument("--GFNModuleConfig",default={'module_name': "NeuralNet",
                                                    'n_hidden_layers': 4,
                                                    'hidden_dim': 256})
optimization.add_argument("--batch_size", type=int, default=128)
optimization.add_argument("--n_iterations", type=int, default=2000)
optimization.add_argument("--device_str",default='cpu',choices=['cpu','cuda'])
# Replay buffer
replay = parser.add_argument_group('Replay Buffer')
replay.add_argument("--replay_buffer_size", type=int, default=0)
# Miscellaneous
misc = parser.add_argument_group('Miscellaneous')
misc.add_argument("--use_wandb", type=bool, default=False)
misc.add_argument("--validation_interval", type=int, default=20)
misc.add_argument("--validation_samples", type=int,default=1000)
args = parser.parse_args()
torch.manual_seed(args.seed)
args.device_str="cpu" if not torch.cuda.is_available() else args.device_str
args.PB_parameterized=True if args.PG_used else  args.PB_parameterized
print('USE_{}_PB_{}_PG_{}'.format(args.device_str,str(args.PB_parameterized),str(args.PG_used)))
env,parametrization,loss_fn=Config(args)
print(loss_fn)
#print(loss_fn.logit_PG)
trajectories_sampler,B_trajectories_sampler=SamplerConfig(env,parametrization)
if args.replay_buffer_size > 0 :
    replay_buffer = ReplayBuffer(env, loss_fn, capacity=args.replay_buffer_size)
else:
    replay_buffer=  None


if args.Loss not in ['TRPO','RL']:
    name=args.Loss+'-B' if args.PB_parameterized else args.Loss+'-U'
else:
    assert args.epsilon_start ==0.0, 'epsilon_start should be 0.0 for on-policy method!'
    if args.Loss in ['TRPO']:
        name= 'RL-T'
    else:
        if args.PG_used:
            name='RL-G'
        else:
            name = args.Loss + '-B' if args.PB_parameterized else args.Loss + '-U'
if args.use_wandb:
    wandb.init(project='DAG_BN_new2')
    wandb.config.update(encode(args))

epsilon=args.epsilon_start
states_visited = 0
for i in trange(args.n_iterations):
    trajectories = trajectories_sampler.sample(n_trajectories=args.batch_size)
    training_samples = trajectories_to_training_samples(trajectories, loss_fn)
    training_last_states=training_samples.last_states
    states_visited += len(trajectories)
    epsilon = args.epsilon_end + (epsilon - args.epsilon_end) * args.epsilon_decay
    trajectories_sampler.actions_sampler.epsilon = epsilon

    training_samples.to_device(args.device_str)
    loss=loss_fn.update_model(training_samples)
    to_log = {"loss": loss.item(), "states_visited": states_visited}
    if len(parametrization.logit_PB.parameters()) and args.Loss in ['RL','TRPO']:
        B_trajectories = B_trajectories_sampler.sample(n_trajectories=128, states=training_last_states)
        B_training_samples = trajectories_to_training_samples(B_trajectories,loss_fn)
        B_training_samples.to_device(args.device_str)
        B_loss = loss_fn.B_update_model_Emp(B_training_samples)
        to_log["B_loss"] = B_loss.item()
    #
    if args.use_wandb: wandb.log(to_log, step=i)
    if (i+1) % args.validation_interval == 0 and i!=0:
        validation_info,_ = validate(env, parametrization, trajectories_sampler,args.validation_samples,exact=True)
        if args.use_wandb:
            wandb.log(validation_info, step=i)
        to_log.update(validation_info)
        tqdm.write(f"{i}: {to_log}")
    if (i+1) % (args.validation_interval*5) == 0 and i != 0:
        parametrization.save_state_dict('./scripts', '{}_{}_'.format(name,i))
        if  args.use_wandb:
            artifact = wandb.Artifact('{}-{}'.format(name,timestamp), type='model')
            artifact.add_file('./scripts/{}_{}_logit_PF.pt'.format(name,i))
            wandb.log_artifact(artifact)
if args.use_wandb: wandb.run.name=name+timestamp
