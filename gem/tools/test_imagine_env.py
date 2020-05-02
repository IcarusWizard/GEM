import argparse
import torch

from gem.envs.utils import make_imagine_env
import matplotlib.pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str)
    args = parser.parse_args()

    env = make_imagine_env(args.checkpoint)

    print('Test imagine mode:')
    obs = torch.rand(1, 3, 64, 64)
    
    state = env.reset(obs)
    action = torch.randn(1, env.action_dim)

    next_state, reward, done, info = env.step(action)
    next_obs = env.render(next_state)

    print(state)
    print(next_state - state)
    print(reward)

    plt.figure()
    plt.imshow(obs[0].permute(1, 2, 0).numpy())
    plt.figure()
    plt.imshow(obs[0].permute(1, 2, 0).numpy())
    plt.show()

    print('Test observation mode:')
    obs = torch.rand(1, 3, 64, 64)
    
    state = env.reset(obs)
    action = torch.randn(1, env.action_dim)

    next_state, reward, done, info = env.step(action, obs=obs)
    next_obs = env.render(next_state)

    print(state)
    print(next_state - state)
    print(reward)

    plt.figure()
    plt.imshow(obs[0].permute(1, 2, 0).numpy())
    plt.figure()
    plt.imshow(obs[0].permute(1, 2, 0).numpy())
    plt.show()