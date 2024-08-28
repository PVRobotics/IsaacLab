# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run an environment with zero action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse
import time
from rlblocks.data_collection.exploration import NormalExplorationTorch
from rlblocks.model.cat_encoder import CatStateEncoder
import torch as t

from rlblocks.model.gaussian_actor import MLPGaussianActor
from rlblocks.model.mlp import MLP

from omni.isaac.lab.app import AppLauncher
from rlblocks.data.basic import Episode
from rlblocks.data_collection.rollout_server import RolloutResponse
from rlblocks.data_collection.rollout_zmq import ZMQRolloutServer

# add argparse arguments
parser = argparse.ArgumentParser(description="Zero agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
import numpy as np

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import parse_env_cfg


def main():
    """Zero actions agent with Isaac Lab environment."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # create environment
    env = gym.make(args_cli.task, render_mode="rgb_array", cfg=env_cfg)
    num_envs = args_cli.num_envs

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")

    env = gym.wrappers.RecordVideo(
        env,
        video_folder='/home/anton/devel/rlblocks/logs/humanoid-isaac-lab/humanoid-3/video',
        step_trigger=lambda step: step % 2000 == 0,
        video_length=400,
    )

    device = 'cuda:0'

    state_len = sum([val.shape[1] for val in env.observation_space.values()])
    action_len = env.action_space.shape[1]
    cat_state_encoder = CatStateEncoder(['policy'])
    action_min = -4
    action_max = 4
    actor = MLPGaussianActor(
        model=MLP(
            input_size=state_len,
            output_size=2 * action_len,
            layers_num=2,
            layer_size=256,
            batch_norm=True,
            scale_last_layer=0.1,
        ),
        action_min=t.tensor(action_min).to(device),
        action_max=t.tensor(action_max).to(device),
        logstd_range=(-4, 0),
        encoder=cat_state_encoder,
    ).to(device)
    actor.eval()

    # reset environment
    states, infos = env.reset()

    # Prepare storage for episodes
    episode_states = [{} for _ in range(num_envs)]
    episode_actions = [[] for _ in range(num_envs)]
    episode_rewards = [[] for _ in range(num_envs)]
    # episode_infos = [[] for _ in range(num_envs)]

    # Initialize state storage for each key in the dict
    for i in range(num_envs):
        for key in states.keys():
            episode_states[i][key] = []

        for key, value in states.items():
            episode_states[i][key].append(value[i].cpu().numpy())

    ep_id = 0

    server = ZMQRolloutServer(address="127.0.0.1", port=5555)
    server.start()

    expl = NormalExplorationTorch(
        std=0.05,
        action_clip=(action_min, action_max),
    )

    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():

            request = server.receive_request()
            print('--- got rollout request')
            actor.load_state_dict(request.actor_state_dict)
            t1 = time.time()
            rollout_eps = []

            for _ in range(request.rollout_len):

                # compute zero actions
                # actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
                actions = actor(states)
                actions[1:] = expl(actions[1:])
                # apply actions
                next_states, rewards, terminated, truncated, infos = env.step(actions)
                # print(f'--- next_states {next_states["policy"].shape}')
                # print(f'--- terminated \n{terminated}')

                # Record the transitions
                for i in range(num_envs):
                    for key, value in next_states.items():
                        episode_states[i][key].append(value[i].cpu().numpy())
                    episode_actions[i].append(actions[i].cpu().numpy())
                    episode_rewards[i].append(rewards[i].cpu().numpy())
                    # episode_infos[i].append(infos[i])

                # Handle termination and truncation
                for i in range(num_envs):
                    if terminated[i] or truncated[i]:
                        # Stack tensors for each key in the states dict
                        episode_stacked_states = {key: np.stack(episode_states[i][key]) for key in episode_states[i].keys()}
                        episode_stacked_states['obs'] = episode_stacked_states['policy']
                        del episode_stacked_states['policy']

                        # Convert lists to tensors
                        episode = Episode(
                            states=episode_stacked_states,
                            actions=np.stack(episode_actions[i]),
                            rewards=np.expand_dims(np.stack(episode_rewards[i]), -1),
                            done=terminated[i],
                            truncated=truncated[i],
                            is_used_exploration=(i != 0),
                            info={},
                            id=f'{i}_{ep_id}'
                        )
                        rollout_eps.append(episode)
                        ep_id += 1
                        # print(episode)
                        # print(episode.shapes)

                        # Process the episode (e.g., save, analyze, etc.)
                        # print(f"Processed Episode {episode.id} with length {len(episodes[i][list(episodes[i].keys())[0]])}")

                        # Reset the environment and episode storage for this environment
                        # reset_state, reset_info = env.reset_at(i)
                        # states[i] = next_states[i]
                        episode_states[i] = {key: [] for key in next_states.keys()}
                        for key, value in next_states.items():
                            episode_states[i][key].append(value[i].cpu().numpy())
                        episode_actions[i] = []
                        episode_rewards[i] = []
                        # episode_infos[i] = []

                states = next_states

            print(f'--- rollout time {time.time() - t1}')
            server.send_rollout(RolloutResponse(episodes=rollout_eps))

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()


# DISPLAY=:0 ./isaaclab.sh -p source/standalone/environments/rlblocks_agent.py --task Isaac-Humanoid-v0 --num_envs 1000
# DISPLAY=:0 ./isaaclab.sh -p source/standalone/environments/rlblocks_agent.py --task Isaac-Humanoid-v0 --num_envs 500 --headless --enable_cameras
