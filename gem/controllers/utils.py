import torch
import random
import numpy as np
from gem.distributions import Onehot, BijectoredDistribution

def world_model_rollout(world_model, controller, reset_obs=None, reset_state=None, horizon=15, mode='train'):
    assert reset_obs is not None or reset_state is not None, "need to provide something to reset the world model"
    rollout_state = []
    rollout_action = []
    rollout_reward = []

    if reset_obs is not None:
        state = world_model.reset(reset_obs)
    else:
        state = world_model.reset_state(reset_state)
    for i in range(horizon):
        action_dist = controller.get_action_dist(state.detach())
        action = action_dist.sample() if mode == 'train' else action_dist.mode()

        next_state, reward, done, info = world_model.step(action)
        rollout_state.append(state)
        rollout_action.append(action)
        rollout_reward.append(reward)

        state = next_state

    rollout_value_dist = [controller.get_critic_dist(state) for state in rollout_state]
    rollout_value = [dist.mode() for dist in rollout_value_dist]

    return rollout_state, rollout_action, rollout_reward, rollout_value, rollout_value_dist

def real_env_rollout(real_env, world_model, controller, action_func=lambda action_dist: action_dist.mode()):
    rollout_state = []
    rollout_obs = []
    rollout_action = []
    rollout_action_entropy = []
    rollout_reward = []
    rollout_predicted_reward = []
    rollout_value = []
    rollout_value_dist = []

    dtype = world_model.dtype
    device = world_model.device

    with torch.no_grad():
        obs = real_env.reset()
        obs = obs['image']
        obs = torch.as_tensor(obs / 255.0 - 0.5, dtype=dtype, device=device).permute(2, 0, 1).unsqueeze(dim=0).contiguous()
        state = world_model.reset(obs)

        while True:
            action_dist, value_dist = controller(state)
            action = action_func(action_dist)
            value = value_dist.mode()

            # step real env
            _action = action[0].cpu().numpy()
            next_obs, reward, done, info = real_env.step(_action)
            next_obs = next_obs['image']
            next_obs = torch.as_tensor(next_obs / 255.0 - 0.5, dtype=dtype, device=device).permute(2, 0, 1).unsqueeze(dim=0).contiguous()

            # step world model
            next_state, predict_reward, _, _ = world_model.step(action, next_obs)

            rollout_state.append(state)
            rollout_obs.append(obs)
            rollout_action.append(action)
            rollout_action_entropy.append(action_dist.entropy())
            rollout_reward.append(reward)
            rollout_predicted_reward.append(predict_reward)
            rollout_value.append(value)
            rollout_value_dist.append(value_dist)

            state = next_state
            obs = next_obs

            if done:
                break
        
        return rollout_state, rollout_obs, rollout_action, rollout_action_entropy, rollout_reward, \
            rollout_predicted_reward, rollout_value, rollout_value_dist

def compute_lambda_return(rewards, values, bootstrap=None, _gamma=0.99, _lambda=0.98):
    next_values = values[1:]
    if bootstrap is None:
        next_values.append(torch.zeros_like(values[-1]))
    else:
        next_values.append(bootstrap)

    g = [rewards[i] + _gamma * (1 - _lambda) * next_values[i] for i in range(len(rewards))]

    lambda_returns = []
    last = next_values[-1]
    for i in reversed(list(range(len(rewards)))):
        last = g[i] + _gamma * _lambda * last
        lambda_returns.append(last)

    return list(reversed(lambda_returns))

def get_explored_action(action_dist, explore_amount=0.3):
    if isinstance(action_dist, BijectoredDistribution):
        # continuous action space
        action = action_dist.sample()
        action = torch.clamp(action + torch.randn_like(action) * explore_amount, -1, 1)
    elif isinstance(action_dist, Onehot):
        # discrete action space
        action = action_dist.sample()
        if random.random() < explore_amount:
            action = torch.zeros_like(action)
            action[np.arange(action.shape[0]), torch.randint(action.shape[1], (action.shape[0],))] = 1.0
    else:
        raise ValueError(f"Distribution type {type(action_dist)} is not support!")

    return action
