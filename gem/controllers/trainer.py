import torch
import numpy as np
from tqdm import tqdm
from tabulate import tabulate
from itertools import chain
from functools import partial
from torch.utils.tensorboard import SummaryWriter

from gem.distributions.utils import stack_normal
from gem.utils import select_gpus, step_loader

from .utils import world_model_rollout, real_env_rollout, compute_lambda_return, get_explored_action

class VGCtTrainer:
    def __init__(self, controller, world_model, test_env, collect_env=None, buffer=None, observation_loader=None, config={}):
        super().__init__()

        self.controller = controller
        self.world_model = world_model
        self.test_env = test_env
        self.collect_env = collect_env
        self.buffer = buffer
        self.observation_loader = observation_loader
        self.config = config

        # config gpus
        select_gpus(self.config['gpu']) 
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.controller = self.controller.to(self.device)
        self.world_model = self.world_model.to(self.device)
        self.dtype = self.world_model.dtype

        # create log writer
        self.writer = SummaryWriter(config['log_name'])

        if self.buffer is not None:
            from gem.envs.wrapper import Collect
            self.observation_iter = self.buffer.generator(self.config['batch_size'], self.config['batch_length'])
        else:
            self.observation_iter = step_loader(self.observation_loader) # used in training

        # config optimizer
        self.actor_optim = torch.optim.Adam(self.controller.get_actor_parameters(), lr=self.config['c_lr'], 
                                      betas=(self.config['c_beta1'], self.config['c_beta2']))

        self.critic_optim = torch.optim.Adam(self.controller.get_critic_parameters(), lr=self.config['c_lr'], 
                                      betas=(self.config['c_beta1'], self.config['c_beta2']))
    
    def train(self):
        for step in tqdm(range(self.config['steps'])):
            if step % self.config['update_step'] == 0:
                self.controller.update_target()

            self.train_step()

            if step % self.config['log_step'] == 0:
                self.log_step(step)

        self.log_step(self.config['steps'])

    def parse_batch(self, batch):
        obs = batch['image'].permute(1, 0, 2, 3, 4).to(self.device).contiguous()
        action = batch['action'].permute(1, 0, 2).to(self.device).contiguous()
        reward = batch['reward'].permute(1, 0).unsqueeze(dim=-1).to(self.device).contiguous() 

        T, B = obs.shape[:2]
        obs = obs.view(T * B, *obs.shape[2:])
        emb = self.world_model.sensor.encode(obs, output_dist=True).mode().view(T, B, -1)

        predictor_loss, prediction, info = self.world_model.predictor(emb, action, reward, use_emb_loss=False)

        states = prediction['state'].detach()
        states = states.view(-1, states.shape[-1])

        return states   

    def train_step(self):
        states = self.parse_batch(next(self.observation_iter))

        # rollout world model
        rollout_state, rollout_action, rollout_reward, rollout_value, rollout_value_dist = \
            world_model_rollout(self.world_model, self.controller, reset_state=states, horizon=self.config['horizon']+1)

        # compute lambda return
        lambda_returns = compute_lambda_return(rollout_reward[:-1], rollout_value[:-1], bootstrap=rollout_value[-1], 
                                            _gamma=self.config['gamma'], _lambda=self.config['lambda'])

        lambda_returns = torch.stack(lambda_returns)
        discont = torch.cat([torch.ones_like(lambda_returns[:1]), self.config['gamma'] * torch.ones_like(lambda_returns[:-1])], dim=0)
        discont = torch.cumprod(discont, dim=0)
        actor_loss = - torch.mean(discont * lambda_returns)
        
        values_dist = stack_normal(rollout_value_dist[:-1])
        critic_loss = - torch.mean(discont * values_dist.log_prob(lambda_returns.detach()))

        info = {
            "actor_loss" : actor_loss.item(),
            "critic_loss" : critic_loss.item(),
            "mean_lambda_return_train" : torch.mean(lambda_returns.detach()).item(),
            "mean_value_train" : torch.mean(torch.stack(rollout_value).detach()).item(),
            "accumulate_reward_train" : torch.sum(torch.stack(rollout_reward).detach()).item() / lambda_returns.shape[1],
        }

        self.actor_optim.zero_grad()
        actor_loss.backward(retain_graph=True)
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(self.controller.get_actor_parameters(), self.config['c_grad_clip'])
        self.actor_optim.step()

        self.critic_optim.zero_grad()
        critic_loss.backward()
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(self.controller.get_critic_parameters(), self.config['c_grad_clip'])
        self.critic_optim.step()

        info.update({
            'actor_grad_norm' : actor_grad_norm,
            "critic_grad_norm" : critic_grad_norm,
        })

        self.last_train_info = info

    def log_step(self, step):
        obs_eval, info_eval = self.test_on_world_model()
        obs_eval_real, pre_obs_eval_real, info_eval_real = self.test_on_real_env()

        info = self.last_train_info
        info.update(info_eval)
        info.update(info_eval_real)
        
        self.writer.add_video('eval_world_model', torch.clamp(obs_eval[:, :16].permute(1, 0, 2, 3, 4) + 0.5, 0, 1), 
            global_step=step, fps=self.config['fps'])

        obs_eval_real = torch.clamp(obs_eval_real.permute(1, 0, 2, 3, 4) + 0.5, 0, 1)
        pre_obs_eval_real = torch.clamp(pre_obs_eval_real.permute(1, 0, 2, 3, 4) + 0.5, 0, 1)
        self.writer.add_video('eval_real', torch.cat([obs_eval_real, pre_obs_eval_real, (pre_obs_eval_real - obs_eval_real + 1) / 2], dim=4), 
            global_step=step, fps=self.config['fps'])

        if self.collect_env is not None:
            obs_collect, pre_obs_collect, info_collect = self.collect_data()

            info.update(info_collect)

            obs_collect = torch.clamp(obs_collect.permute(1, 0, 2, 3, 4) + 0.5, 0, 1)
            pre_obs_collect = torch.clamp(pre_obs_collect.permute(1, 0, 2, 3, 4) + 0.5, 0, 1)
            self.writer.add_video('collection', torch.cat([obs_collect, pre_obs_collect, (pre_obs_collect - obs_collect + 1) / 2], dim=4), 
                global_step=step, fps=self.config['fps'])

        print('In Step {}'.format(step))
        for k, v in info.items():
            self.writer.add_scalar('controller/' + k, v, global_step=step)

        print(tabulate(info.items(), numalign="right"))
        
    def test_on_world_model(self):
        states = self.parse_batch(next(self.observation_iter))

        with torch.no_grad():
            # rollout world model
            rollout_state, rollout_action, rollout_reward, rollout_value, rollout_value_dist = \
                world_model_rollout(self.world_model, self.controller, reset_state=states, horizon=self.config['horizon']+1, mode='test')

            lambda_returns = compute_lambda_return(rollout_reward[:-1], rollout_value[:-1], bootstrap=rollout_value[-1], 
                                    _gamma=self.config['gamma'], _lambda=self.config['lambda'])

            rollout_state = torch.stack(rollout_state)
            rollout_obs = self.world_model.render(rollout_state)

            info = {
                "mean_lambda_return_eval" : torch.mean(torch.stack(lambda_returns).detach()).item(),
                "mean_value_eval" : torch.mean(torch.stack(rollout_value)).item(),
                "accumulate_reward_eval" : torch.sum(torch.stack(rollout_reward)).item() / lambda_returns[0].shape[0],
            }

            return rollout_obs, info
    
    def test_on_real_env(self):
        # rollout real world
        rollout_state, rollout_obs, rollout_action, rollout_action_entropy, rollout_reward, \
                rollout_predicted_reward, rollout_value, rollout_value_dist = real_env_rollout(self.test_env, self.world_model, self.controller)

        rollout_obs = torch.stack(rollout_obs)
        rollout_state = torch.stack(rollout_state)
        rollout_predicted_obs = self.world_model.render(rollout_state)

        info = {
            "accumulate_reward_eval_real" : torch.sum(torch.stack(rollout_predicted_reward)).item(),
            "accumulate_real_reward_eval_real" : np.sum(rollout_reward),
            "mean_action_entropy_eval_real" : torch.mean(torch.stack(rollout_action_entropy)).item(),
            "mean_value_eval_real" : torch.mean(torch.stack(rollout_value)).item(),
        }

        return rollout_obs, rollout_predicted_obs, info

    def collect_data(self):
        # rollout real world
        action_func = partial(get_explored_action, explore_amount=self.config['explore_amount'])
        rollout_state, rollout_obs, rollout_action, rollout_action_entropy, rollout_reward, \
                rollout_predicted_reward, rollout_value, rollout_value_dist = real_env_rollout(
                    self.collect_env, self.world_model, self.controller, action_func)

        rollout_obs = torch.stack(rollout_obs)
        rollout_state = torch.stack(rollout_state)
        rollout_predicted_obs = self.world_model.render(rollout_state)

        info = {
            "accumulate_reward_collection" : torch.sum(torch.stack(rollout_predicted_reward)).item(),
            "accumulate_real_reward_collection" : np.sum(rollout_reward),
            "mean_action_entropy_collection" : torch.mean(torch.stack(rollout_action_entropy)).item(),
            "mean_value_eval_collection" : torch.mean(torch.stack(rollout_value)).item(),
        }

        return rollout_obs, rollout_predicted_obs, info

    def save(self, filename):
        torch.save({
            "controller_state_dict" : self.controller.state_dict(),
            "actor_optimizer_state_dict" : self.actor_optim.state_dict(),
            "critic_optimizer_state_dict" : self.critic_optim.state_dict(),
            "config" : self.config,
            "controller_parameters" : self.config['controller_param'],
            "seed" : self.config['seed'],
        }, filename)

    def restore(self, checkpoint):
        self.controller.load_state_dict(checkpoint['controller_state_dict'])
        self.controller = self.controller.to(self.device) # make sure model on right device
        self.actor_optim.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optim.load_state_dict(checkpoint['critic_optimizer_state_dict'])