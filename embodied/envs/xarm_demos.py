import collections
import dataclasses
import enum
import re
import socket
import sys
import time
import traceback
import warnings
from typing import Any, Dict, Optional, Tuple, Union

import cv2
import dcargs
import numpy as np
import pyrealsense2 as rs

import embodied
from embodied.envs.spacemouse import SpaceMouse


class SpaceMouseAgent:
    def __init__(self) -> None:
        self._mouse = SpaceMouse()

    def act(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        while True:
            input_position = np.zeros((2,))
            if np.linalg.norm(self._mouse.input_pos) > 10:
                # print(self._mouse.input_pos)
                input_position = (
                    np.asarray(
                        [
                            self._mouse.input_pos[1],
                            self._mouse.input_pos[0],
                            -self._mouse.input_pos[2],
                        ]
                    )
                    / 350
                )
                input_position = input_position[:2]

            if np.abs(input_position[0]) > np.abs(input_position[1]):
                if input_position[0] > 0.2:
                    return {"action": 0}
                elif input_position[0] < -0.2:
                    return {"action": 1}
            else:
                if input_position[1] > 0.2:
                    return {"action": 2}
                elif input_position[1] < -0.2:
                    return {"action": 3}

            if self._mouse.input_button1:
                return {"action": 4}
            if self._mouse.input_button0:
                return {"action": 5}

            time.sleep(0.005)  # wait for an action


def collect_xarm_demos():

    args = embodied.Flags({"outdir": "/dev/null", "steps": 1e6, "length": 100,}).parse()

    logger = embodied.Logger(embodied.Counter(), [
        embodied.logger.TerminalOutput(),
        embodied.logger.JSONLOutput(args.outdir, 'metrics.jsonl'),
        embodied.logger.TensorBoardOutput(args.outdir),
    ], multiplier=1)

    def on_episode(ep, _):
        score = np.sum(ep['reward'])
        length = len(ep['reward'])
        logger.add({
          'score': score,
          'length': length,
          'avg_reward': score / length,
        })
        logger.write(fps=True)

    score = 0
    length = 0

    outdir = embodied.Path(args.outdir)
    store = embodied.replay.DiskStore(outdir, 1e6, parallel=True)
    replay = embodied.replay.FixedLength(store)

    env = embodied.envs.load_env("ur5_real", length=args.length)
    driver = embodied.Driver(env)
    driver.on_step(replay.add)
    driver.on_episode(on_episode)

    agent = SpaceMouseAgent()

    def policy(obs, state=None, mode="train"):

        # cv2.imshow("img", obs["image"])
        # cv2.waitKey(1)

        act = agent.act(obs)
        index = act['action']
        act['action'] = np.zeros((1, 6,))
        act['action'][0, index] = 1
        act['reset'] = obs['is_last']
        logger.step.increment()
        return act, state

    driver(policy, steps=args.steps)


if __name__ == "__main__":
    collect_xarm_demos()
