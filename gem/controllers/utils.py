import torch

def world_model_rollout(world_model, controller, reset_obs, horizon=50, mode='train'):
    rollout_state = []
    rollout_action = []
    rollout_reward = []

    state = world_model.reset(reset_obs)
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

def real_env_rollout(real_env, world_model, controller):
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
            action = action_dist.mode()
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