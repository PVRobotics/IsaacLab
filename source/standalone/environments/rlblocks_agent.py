# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run an environment with zero action agent."""

"""Launch Isaac Sim Simulator first."""

import os

from rlblocks.data.rollout_buffer import RolloutBuffer
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import time
from typing import Dict
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
        video_folder='/home/anton/devel/rlblocks/logs/humanoid-isaac-lab/humanoid-13/video',
        step_trigger=lambda step: step % 3000 == 0,
        video_length=400,
    )

    device = 'cuda:0'

    state_len = sum([val.shape[1] for val in env.observation_space.values()])
    action_len = env.action_space.shape[1]
    cat_state_encoder = CatStateEncoder(['policy'])
    action_min = -1.5
    action_max = 1.5
    actor = MLPGaussianActor(
        model=MLP(
            input_size=state_len,
            output_size=2 * action_len,
            layers_num=2,
            layer_size=256,
            batch_norm=True,
            scale_last_layer=0.01,
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
    rollout_buffer = RolloutBuffer(args_cli.num_envs)
    rollout_buffer.store_first_obs(states)

    server = ZMQRolloutServer(address="127.0.0.1", port=5555)
    server.start()

    expl = NormalExplorationTorch(
        std=0.1,
        action_clip=(action_min, action_max),
        repeat=1,
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

                # compute actions
                actions, info = actor(states, add_info=True)
                actions[:1] = info['mu'][:1]
                actions[1:] = expl(actions[1:])

                # apply actions
                next_states, rewards, terminated, truncated, infos = env.step(actions)
                # print(f'--- next_states {next_states["policy"].shape}')
                # print(f'--- terminated \n{terminated}')

                # Record the transitions
                rollout_buffer.store_transitions(
                    states=next_states,
                    actions=actions,
                    rewards=rewards,
                )

                # Handle termination and truncation
                for ind in range(num_envs):
                    if terminated[ind] or truncated[ind]:

                        episode = rollout_buffer.get_episode(
                            ind=ind,
                            terminated=bool(terminated[ind]),
                            truncated=bool(truncated[ind]),
                            is_used_exploration=(ind != 0),
                            partial=False,
                        )
                        episode.states['obs'] = episode.states['policy']
                        del episode.states['policy']
                        rollout_eps.append(episode)
                        rollout_buffer.reset_episode(
                            ind=ind,
                            states=next_states,
                        )
                        rollout_buffer.inc_ep_ind(ind)
                        # print(f'--- Done {episode}')

                states = next_states

            # rollout steps ended, send also not finished episodes
            for ind in range(num_envs):
                if rollout_buffer.get_episode_len(ind) > 0:
                    episode = rollout_buffer.get_episode(
                        ind=ind,
                        terminated=False,
                        truncated=False,
                        is_used_exploration=(ind != 0),
                        partial=True,
                    )
                    episode.states['obs'] = episode.states['policy']
                    del episode.states['policy']
                    rollout_eps.append(episode)
                    rollout_buffer.reset_episode(
                        ind=ind,
                        states=next_states,
                    )
                    # print(f'---      {episode}')

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
