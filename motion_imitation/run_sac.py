# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)

import argparse
import numpy as np
import os
import random
import tensorflow.compat.v1 as tf
import time

from motion_imitation.envs import env_builder as env_builder
from motion_imitation.utilities import motion_util
import sac_dev.learning.sac_agent as sac_agent
import sac_dev.util.mpi_util as mpi_util
import sac_dev.sac_configs
from motion_imitation.envs.env_wrappers import imitation_task
from motion_imitation.envs.env_wrappers import logging_wrapper
from safe_outdoor import resetters
from safe_outdoor.learning import yielding_sac_agent

ENABLE_ENV_RANDOMIZER = True

def find_file(filename):
    if os.path.isfile(filename):
        return filename
    possible_filepath = os.path.join(currentdir, filename)
    if os.path.isfile(possible_filepath):
        return possible_filepath
    possible_filepath = os.path.join(parentdir, filename)
    if os.path.isfile(possible_filepath):
        return possible_filepath
    raise ValueError("No such file: '{}'".format(filename))

def set_rand_seed(seed=None):
    if seed is None:
        seed = int(time.time())

    seed += 97 * mpi_util.get_proc_rank()

    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    return

def enable_gpus(gpu_str):
    if (gpu_str is not ""):
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    return

def build_agent(env, variant, agent_cls=sac_agent.SACAgent):
    graph = tf.Graph()
    sess = tf.Session(graph=graph)
    agent = agent_cls(env=env, sess=sess, **variant)

    return agent

def train_multitask(
    env,
    total_timesteps,
    output_dir,
    int_save_freq,
    test_episodes,
    variant,
    forward_model_file,
    reset_func,
    backward_motion_file="",
    backward_model_file="",
):
    forward_task = env.task
    task_dict = {"Forward": forward_task}
    model_file_dict = {"Forward": forward_model_file}

    if backward_motion_file != "":
        backward_task = imitation_task.ImitationTask(
                        ref_motion_filenames=[find_file(backward_motion_file)],
                        enable_cycle_sync=forward_task.cycle_sync_enabled,
                        tar_frame_steps=forward_task.tar_frame_steps,
                        ref_state_init_prob=forward_task.ref_state_init_prob,
                        enable_rand_init_time=forward_task.enable_rand_init_time,
                        warmup_time=forward_task.warmup_time)
        task_dict["Backward"] = backward_task
        model_file_dict["Backward"] = backward_model_file

    train_funcs = {}
    for task_name, model_file in model_file_dict.items():
        my_variant = variant.copy()

        model = build_agent(env, my_variant, yielding_sac_agent.YieldingSACAgent)
        if model_file != "":
            model.load_model(model_file)
        
        train_funcs[task_name] = model.train(
            max_samples=total_timesteps,
            test_episodes=test_episodes,
            output_dir=os.path.join(output_dir, task_name),
            output_iters=int_save_freq,
            variant=my_variant)
    direction_label = -1
    counter = 0
    task_name = "Forward"
    while True:
        reset_func()
        if backward_motion_file != "":
            pos = env.robot.GetBasePosition()
            init_mat = motion_util.to_matrix(pos,
                                             env.robot.GetTrueBaseRollPitchYaw())
            init_mat_inv = np.linalg.inv(init_mat)
            local_ox, local_oy = init_mat_inv[0:2, 2]
            if local_ox >= 0.0:
                task_name = "Forward"
            else:
                task_name = "Backward"

            env.pybullet_client.removeUserDebugItem(direction_label)
            direction_label = env.pybullet_client.addUserDebugText(
                                  task_name, (0, 0, 0.1), (1, 1, 1),
                                  parentObjectUniqueId=env.robot.quadruped)

        env.set_task(task_dict[task_name])

        try:
            next(train_funcs[task_name])
        except StopIteration:
            return

def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--seed", dest="seed", type=int, default=None)
    arg_parser.add_argument("--mode", dest="mode", type=str, default="train")
    arg_parser.add_argument("--motion_file", dest="motion_file", type=str, default="motion_imitation/data/motions/pace.txt")
    arg_parser.add_argument("--backward_motion_file", dest="backward_motion_file", type=str, default="")
    arg_parser.add_argument("--visualize", dest="visualize", action="store_true", default=False)
    # Root output directory. An additional subdir with the datetime is added.
    arg_parser.add_argument("--output_dir", dest="output_dir", type=str, default="output")
    # Optionally add a descriptive suffix to the datetime.
    arg_parser.add_argument("--output_suffix", dest="output_suffix", type=str, default="")
    arg_parser.add_argument("--num_test_episodes", dest="num_test_episodes", type=int, default=1)
    arg_parser.add_argument("--model_file", dest="model_file", type=str, default="")
    arg_parser.add_argument("--backward_model_file", dest="backward_model_file", type=str, default="")
    arg_parser.add_argument("--getup_model_file", dest="getup_model_file", type=str, default="motion_imitation/data/policies/model-004050.ckpt")
    arg_parser.add_argument("--total_timesteps", dest="total_timesteps", type=int, default=2e8)
    arg_parser.add_argument("--int_save_freq", dest="int_save_freq", type=int, default=50) # save intermediate model every n policy steps
    arg_parser.add_argument("--gpu", dest="gpu", default="")
    arg_parser.add_argument("--train_reset", dest="train_reset", action="store_true", default=False)
    arg_parser.add_argument("--finetune", dest="finetune", action="store_true", default=False)
    arg_parser.add_argument("--use_redq", dest="use_redq", action="store_true", default=False)
    arg_parser.add_argument("--multitask", dest="multitask", action="store_true")
    arg_parser.add_argument("--no_multitask", dest="multitask", action="store_false")
    arg_parser.set_defaults(multitask=False)
    arg_parser.add_argument("--real_robot", dest="real", action="store_true")
    arg_parser.add_argument("--sim_robot", dest="real", action="store_false")
    arg_parser.set_defaults(real=False)
    arg_parser.add_argument("--realistic_sim", dest="realistic_sim", action="store_true", default=False)
    arg_parser.add_argument("--mocap_grpc_server", dest="mocap_grpc_server", type=str, default=None)
    arg_parser.add_argument("--no_env_logging", dest="env_logging", action="store_false", default=True)

    args = arg_parser.parse_args()

    tf.logging.set_verbosity(tf.logging.ERROR)

    agent_configs = {}
    env_id = "A1-Motion-Imitation"
    if args.use_redq:
        env_id += "-REDQ"
    else:
        env_id += "-Vanilla-SAC"

    if args.finetune:
        env_id += "-Finetune"
    else:
        env_id += "-Pretrain"

    agent_configs = sac_dev.sac_configs.SAC_CONFIGS[env_id]

    # Quick run to make sure bits are flowing.
    if args.mode == "canary":
        args.int_save_freq = 1
        args.num_test_episodes = 1
        args.output_dir = "/tmp/safe_outdoor_canary_runs"
        args.total_timesteps = 1
        agent_configs["init_samples"] = 10

    num_procs = mpi_util.get_num_procs()

    enable_gpus(args.gpu)
    set_rand_seed(int(time.time()))
    suf = "_" + args.output_suffix if args.output_suffix else ""
    output_dir = os.path.join(args.output_dir,
                              time.strftime("%Y-%m-%d_%H%M_%S", time.localtime()) + suf)

    enable_env_rand = ENABLE_ENV_RANDOMIZER and (args.mode != "test")

    env = env_builder.build_env("reset" if args.train_reset else "imitate",
                                motion_files=[find_file(args.motion_file)],
                                num_parallel_envs=num_procs,
                                mode=args.mode,
                                enable_randomizer=enable_env_rand,
                                enable_rendering=args.visualize,
                                use_real_robot=args.real,
                                reset_at_current_position=args.multitask,
                                realistic_sim=args.realistic_sim)
    if args.env_logging:
      env = logging_wrapper.LoggingWrapper(env, output_dir,
                                           args.mocap_grpc_server)

    if args.multitask:
        if args.mode == "test":
            raise NotImplementedError("Only training implemented for multitask")
        if args.real or args.realistic_sim:
            resetter = resetters.GetupResetter(env, args.getup_model_file)
        else:
            resetter = lambda: None
        train_multitask(env=env,
                        backward_motion_file=args.backward_motion_file,
                        total_timesteps=int(args.total_timesteps),
                        output_dir=output_dir,
                        int_save_freq=args.int_save_freq,
                        test_episodes=args.num_test_episodes,
                        variant=agent_configs,
                        forward_model_file=args.model_file,
                        backward_model_file=args.backward_model_file,
                        reset_func=resetter)
        return

    if args.mode == "standup":
        resetter = resetters.GetupResetter(env, args.getup_model_file)
        for i in range(args.num_test_episodes):
            input("Strike <Enter> to stand.")
            resetter()
            input("Strike <Enter> to fall.")
            env.robot.Brake()
        return

    model = build_agent(env, variant=agent_configs)

    if args.model_file != "":
        model.load_model(args.model_file)

    if args.mode in ("train", "canary"):
        model.train(max_samples=args.total_timesteps,
                    test_episodes=args.num_test_episodes,
                    output_dir=output_dir,
                    output_iters=args.int_save_freq,
                    variant=agent_configs)
    elif args.mode == "test":
        model.eval(num_episodes=args.num_test_episodes)
    else:
        assert False, "Unsupported mode: " + args.mode

    return

if __name__ == '__main__':
  main()
