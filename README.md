# DayDreamer: World Models for Physical Robot Learning

Official implementation of the [DayDreamer][paper] algorithm in TensorFlow 2.

![DayDreamer Robots](https://github.com/danijar/daydreamer/raw/main/media/header.gif)

If you find this code useful, please reference in your paper:

```
@article{wu2022daydreamer,
  title={DayDreamer: World Models for Physical Robot Learning},
  author={Wu, Philipp and Escontrela, Alejandro and Hafner, Danijar and Goldberg, Ken and Abbeel, Pieter},
  journal={Conference on Robot Learning},
  year={2022}
}
```

[paper]: https://danijar.com/daydreamer/

## Method

DayDreamer learns a world model and an actor critic behavior to train robots
from small amounts of experience in the real world, without using simulators.
At a high level, DayDreamer consists of two processes. The actor process
interacts with the environment and stores experiences into the replay buffer.
The learner samples data from the replay buffer to train the world model, and
then uses imagined predictions of the world model to train the behavior.

![DayDreamer Model](https://github.com/danijar/daydreamer/raw/main/media/model.png)

To learn from proprioceptive and visual inputs alike, the world model fuses the
sensory inputs of the same time step together into a compact discrete
representation. A recurrent neural network predicts the sequence of these
representations given actions. From the resulting recurrent states and
representations, DayDreamer reconstructs its inputs and predicts rewards and
episode ends.

Given the world model, the actor critic learns farsighted behaviors using
on-policy reinforcement learning purely inside the representation space of the
world model.

For more information:

- [Project website](https://danijar.com/project/daydreamer/)
- [Research paper](https://arxiv.org/pdf/2206.14176.pdf)
- [YouTube video](https://www.youtube.com/watch?v=xAXvfVTgqr0)

## Setup

```
pip install tensorflow tensorflow_probability ruamel.yaml cloudpickle
```

## Instructions

To run DayDreamer, open two terminals to execute the commands for the learner
bnd the actor in parallel. To view metrics, point TensorBoard at the log
directory. For more information, also see the [DreamerV2][dv3] repository.

[dv3]: https://github.com/danijar/dreamerv2

A1 Robot:

```
rm -rf ~/logdir/run1
```

```
CUDA_VISIBLE_DEVICES=0 python embodied/agents/dreamerv2plus/train.py --configs a1 --task a1_sim --run learning --tf.platform gpu --logdir ~/logdir/run1
```

```
CUDA_VISIBLE_DEVICES=1 python embodied/agents/dreamerv2plus/train.py --configs a1 --task a1_real --run acting --tf.platform gpu --env.kbreset True --imag_horizon 1 --replay_chunk 8 --replay_fixed.minlen 32 --imag_horizon 1 --logdir ~/logdir/run1
```

XArm Robot:

```
rm -rf ~/logdir/run1
```

```
CUDA_VISIBLE_DEVICES=0 python embodied/agents/dreamerv2plus/train.py --configs xarm --run learning --task xarm_dummy --tf.platform gpu --logdir ~/logdir/run1
```

```
CUDA_VISIBLE_DEVICES=-1 python embodied/agents/dreamerv2plus/train.py --configs xarm --run acting --task xarm_real --env.kbreset True --tf.platform cpu --tf.jit False --logdir ~/logdir/run1
```

UR5 Robot:

```
rm -rf ~/logdir/run1
```

```
CUDA_VISIBLE_DEVICES=0 python embodied/agents/dreamerv2plus/train.py --configs ur5 --run learning --task ur5_dummy --tf.platform gpu --logdir ~/logdir/run11
```

```
CUDA_VISIBLE_DEVICES=1 python embodied/agents/dreamerv2plus/train.py --configs ur5 --run acting --task ur5_real --env.kbreset True --tf.platform cpu --tf.jit False --logdir ~/logdir/run11
```

## Questions

Please [open an issue][issues] on Github.

[issues]: https://github.com/danijar/daydreamer/issues
