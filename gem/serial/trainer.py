import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from itertools import chain

from gem.utils import nats2bits, select_gpus
from gem.distributions.utils import stack_normal
from gem.controllers.utils import compute_lambda_return, world_model_rollout, real_env_rollout

class SerialAgentTrainer:
    def __init__(self, world_model, controller, test_env, collect_env, buffer, config={}):
        self.world_model = world_model
        self.controller = controller
        self.test_env = test_env
        self.collect_env = collect_env
        self.buffer = buffer
        self.config = config

        # config gpus
        select_gpus(self.config['gpu']) 
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.world_model = self.world_model.to(self.device)
        self.controller = self.controller.to(self.device)
        self.sensor = self.world_model.sensor
        self.predictor = self.world_model.predictor

        # create log writer
        self.writer = SummaryWriter(config['log_name'])

        self.data_iter = self.buffer.generator(config['batch_size'], config['batch_length'])

        # config optimizer
        self.model_optim = torch.optim.Adam(chain(self.sensor.parameters(), self.predictor.parameters()), lr=self.config['m_lr'], 
                                            betas=(self.config['m_beta1'], self.config['m_beta2']))
        
        self.actor_optim = torch.optim.Adam(self.controller.get_actor_parameters(), lr=self.config['c_lr'], 
                                            betas=(self.config['c_beta1'], self.config['c_beta2']))

        self.critic_optim = torch.optim.Adam(self.controller.get_critic_parameters(), lr=self.config['c_lr'], 
                                             betas=(self.config['c_beta1'], self.config['c_beta2']))

    def train(self):
        for step in tqdm(range(self.config['steps'])):
            self.train_step()

            if step % self.config['log_step'] == 0:
                self.log_step(step)
        self.log_step(self.config['steps'])

    def get_loss_info(self, batch):
        obs, action, reward = self.parse_batch(batch)

        T, B = obs.shape[:2]
        obs = obs.view(T * B, *obs.shape[2:])
        emb = self.sensor.encode(obs, output_dist=True).mode().view(T, B, -1)

        predictor_loss, prediction, info = self.predictor(emb, action, reward, use_emb_loss=False)

        pre_emb = prediction['emb'].view(T * B, -1)
        pre_obs_dist = self.sensor.decode(pre_emb, output_dist=True)
        reconstruction_loss = - pre_obs_dist.log_prob(obs)
        reconstruction_loss = torch.mean(torch.sum(reconstruction_loss, dim=(1, 2, 3)))

        model_loss = reconstruction_loss + predictor_loss
        info.update({
            "model_loss" : model_loss.item(),
            "renconstruction_loss" : reconstruction_loss.item(),
        })

        states = prediction['state'].detach()
        states = states.view(-1, states.shape[-1])

        # rollout world model
        rollout_state, rollout_action, rollout_reward, rollout_value, rollout_value_dist = \
            world_model_rollout(self.world_model, self.controller, reset_state=states, horizon=self.config['horizon']+1)

        # compute lambda return
        lambda_returns = compute_lambda_return(rollout_reward[:-1], rollout_value[:-1], bootstrap=rollout_value[-1], 
                                            _gamma=self.config['gamma'], _lambda=self.config['lambda'])

        lambda_returns = torch.stack(lambda_returns)
        actor_loss = - torch.mean(lambda_returns)
        
        values_dist = stack_normal(rollout_value_dist[:-1])
        critic_loss = - torch.mean(values_dist.log_prob(lambda_returns.detach()))

        info.update({
            "actor_loss" : actor_loss.item(),
            "critic_loss" : critic_loss.item(),
            "mean_lambda_return_train" : torch.mean(lambda_returns.detach()).item(),
            "mean_value_train" : torch.mean(torch.stack(rollout_value).detach()).item(),
            "accumulate_reward_train" : torch.sum(torch.stack(rollout_reward).detach()).item() / obs.shape[0],
        })

        return model_loss, actor_loss, critic_loss, info
    
    def train_step(self):
        batch = next(self.data_iter)

        model_loss, actor_loss, critic_loss, info = self.get_loss_info(batch)

        self.model_optim.zero_grad()
        model_loss.backward()
        model_grad_norm = torch.nn.utils.clip_grad_norm_(chain(self.sensor.parameters(), self.predictor.parameters()), self.config['m_grad_clip'])
        self.model_optim.step()

        self.actor_optim.zero_grad()
        actor_loss.backward(retain_graph=True)
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(self.controller.get_actor_parameters(), self.config['c_grad_clip'])
        self.actor_optim.step()

        self.critic_optim.zero_grad()
        critic_loss.backward()
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(self.controller.get_critic_parameters(), self.config['c_grad_clip'])
        self.critic_optim.step()

        info.update({
            "model_grad_norm" : model_grad_norm,
            'actor_grad_norm' : actor_grad_norm,
            "critic_grad_norm" : critic_grad_norm,
        })

        self.last_train_info = info

    def log_step(self, step):
        obs_eval, info_eval = self.test_on_world_model()
        obs_eval_real, pre_obs_eval_real, info_eval_real = self.test_on_real_env()
        obs_collect, pre_obs_collect, info_collect = self.collect_data()

        print('In Step {}'.format(step))
        print('-' * 15)
        for k, v in self.last_train_info.items():
            print('{0} is {1:{2}}'.format(k, v, '.2f'))
            self.writer.add_scalar('serial_agent/' + k, v, global_step=step)
        for k, v in info_eval.items():
            print('{0} is {1:{2}}'.format(k, v, '.2f'))
            self.writer.add_scalar('serial_agent/' + k, v, global_step=step)
        for k, v in info_eval_real.items():
            print('{0} is {1:{2}}'.format(k, v, '.2f'))
            self.writer.add_scalar('serial_agent/' + k, v, global_step=step)
        for k, v in info_collect.items():
            print('{0} is {1:{2}}'.format(k, v, '.2f'))
            self.writer.add_scalar('serial_agent/' + k, v, global_step=step)
        
        self.writer.add_video('eval_world_model', torch.clamp(obs_eval.permute(1, 0, 2, 3, 4) + 0.5, 0, 1), 
            global_step=step, fps=self.config['fps'])

        obs_eval_real = torch.clamp(obs_eval_real.permute(1, 0, 2, 3, 4) + 0.5, 0, 1)
        pre_obs_eval_real = torch.clamp(pre_obs_eval_real.permute(1, 0, 2, 3, 4) + 0.5, 0, 1)
        self.writer.add_video('eval_real', torch.cat([obs_eval_real, pre_obs_eval_real, (pre_obs_eval_real - obs_eval_real + 1) / 2], dim=4), 
            global_step=step, fps=self.config['fps'])

        obs_collect = torch.clamp(obs_collect.permute(1, 0, 2, 3, 4) + 0.5, 0, 1)
        pre_obs_collect = torch.clamp(pre_obs_collect.permute(1, 0, 2, 3, 4) + 0.5, 0, 1)
        self.writer.add_video('collection', torch.cat([obs_collect, pre_obs_collect, (pre_obs_collect - obs_collect + 1) / 2], dim=4), 
            global_step=step, fps=self.config['fps'])

        self.writer.flush()

    def parse_batch(self, batch):
        obs = batch['image'].permute(1, 0, 2, 3, 4).to(self.device).contiguous()
        action = batch['action'].permute(1, 0, 2).to(self.device).contiguous()
        reward = batch['reward'].permute(1, 0).unsqueeze(dim=-1).to(self.device).contiguous() if self.config['predict_reward'] else None  
        return obs, action, reward    

    def test_on_world_model(self):
        obs = self.buffer.sample_image(16)[0].to(self.device)
        with torch.no_grad():
            # rollout world model
            rollout_state, rollout_action, rollout_reward, rollout_value, rollout_value_dist = \
                world_model_rollout(self.world_model, self.controller, reset_obs=obs, horizon=self.config['horizon']+1, mode='test')

            lambda_returns = compute_lambda_return(rollout_reward[:-1], rollout_value[:-1], bootstrap=rollout_value[-1], 
                                    _gamma=self.config['gamma'], _lambda=self.config['lambda'])

            rollout_state = torch.stack(rollout_state)
            rollout_obs = self.world_model.render(rollout_state)

            info = {
                "mean_lambda_return_eval" : torch.mean(torch.stack(lambda_returns).detach()).item(),
                "mean_value_eval" : torch.mean(torch.stack(rollout_value)).item(),
                "accumulate_reward_eval" : torch.sum(torch.stack(rollout_reward)).item() / obs.shape[0],
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
        action_func = lambda action_dist: action_dist.sample()
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
            "sensor_state_dict" : self.sensor.state_dict(),
            "predictor_state_dict" : self.predictor.state_dict(),
            "controller_state_dict" : self.controller.state_dict(),
            "model_optimizer_state_dict" : self.model_optim.state_dict(),
            "actor_optimizer_state_dict" : self.actor_optim.state_dict(),
            "critic_optimizer_state_dict" : self.critic_optim.state_dict(),
            "config" : self.config,
            "sensor_parameters" : self.config['sensor_param'],
            "predictor_parameters" : self.config['predictor_param'],
            "controller_parameters" : self.config['controller_param'],
            "seed" : self.config['seed'],
        }, filename)

    def restore(self, filename):
        checkpoint = torch.load(filename, map_location='cpu')
        self.sensor.load_state_dict(checkpoint['sensor_state_dict'])
        self.predictor.load_state_dict(checkpoint['predictor_state_dict'])
        self.controller.load_state_dict(checkpoint['controller_state_dict'])
        self.world_model = self.world_model.to(self.device)
        self.controller = self.controller.to(self.device)
        self.model_optim.load_state_dict(checkpoint['model_optimizer_state_dict'])
        self.actor_optim.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optim.load_state_dict(checkpoint['critic_optimizer_state_dict'])