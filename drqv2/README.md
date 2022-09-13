

# DrQ-v2: Improved Data-Augmented RL Agent

This is an original PyTorch implementation of DrQ-v2 from

[[Mastering Visual Continuous Control: Improved Data-Augmented Reinforcement Learning]](https://arxiv.org/abs/2107.09645) by

[Denis Yarats](https://cs.nyu.edu/~dy1042/), [Rob Fergus](https://cs.nyu.edu/~fergus/pmwiki/pmwiki.php), [Alessandro Lazaric](http://chercheurs.lille.inria.fr/~lazaric/Webpage/Home/Home.html), and [Lerrel Pinto](https://www.lerrelpinto.com).

<p align="center">
  <img width="19.5%" src="https://i.imgur.com/NzY7Pyv.gif">
  <img width="19.5%" src="https://imgur.com/O5Va3NY.gif">
  <img width="19.5%" src="https://imgur.com/PCOR9Mm.gif">
  <img width="19.5%" src="https://imgur.com/H0ab6tz.gif">
  <img width="19.5%" src="https://imgur.com/sDGgRos.gif">
  <img width="19.5%" src="https://imgur.com/gj3qo1X.gif">
  <img width="19.5%" src="https://imgur.com/FFzRwFt.gif">
  <img width="19.5%" src="https://imgur.com/W5BKyRL.gif">
  <img width="19.5%" src="https://imgur.com/qwOGfRQ.gif">
  <img width="19.5%" src="https://imgur.com/Uubf00R.gif">
 </p>

## Method
DrQ-v2 is a model-free off-policy algorithm for image-based continuous control. DrQ-v2 builds on [DrQ](https://github.com/denisyarats/drq), an actor-critic approach that uses data augmentation to learn directly from pixels. We introduce several improvements including:
- Switch the base RL learner from SAC to DDPG.
- Incorporate n-step returns to estimate TD error.
- Introduce a decaying schedule for exploration noise.
- Make implementation 3.5 times faster.
- Find better hyper-parameters.

<p align="center">
  <img src="https://i.imgur.com/SemY10G.png" width="100%"/>
</p>

These changes allow us to significantly improve sample efficiency and wall-clock training time on a set of challenging tasks from the [DeepMind Control Suite](https://github.com/deepmind/dm_control) compared to prior methods. Furthermore, DrQ-v2 is able to solve complex humanoid locomotion tasks directly from pixel observations, previously unattained by model-free RL.

<p align="center">
  <img width="100%" src="https://imgur.com/mrS4fFA.png">
  <img width="100%" src="https://imgur.com/pPd1ks6.png">
 </p>

## Citation

If you use this repo in your research, please consider citing the paper as follows:
```
@article{yarats2021drqv2,
  title={Mastering Visual Continuous Control: Improved Data-Augmented Reinforcement Learning},
  author={Denis Yarats and Rob Fergus and Alessandro Lazaric and Lerrel Pinto},
  journal={arXiv preprint arXiv:2107.09645},
  year={2021}
}
```
Please also cite our original paper:
```
@inproceedings{yarats2021image,
  title={Image Augmentation Is All You Need: Regularizing Deep Reinforcement Learning from Pixels},
  author={Denis Yarats and Ilya Kostrikov and Rob Fergus},
  booktitle={International Conference on Learning Representations},
  year={2021},
  url={https://openreview.net/forum?id=GY6-6sTvGaf}
}
```

## Instructions

Install [MuJoCo](http://www.mujoco.org/) if it is not already the case:

* Obtain a license on the [MuJoCo website](https://www.roboti.us/license.html).
* Download MuJoCo binaries [here](https://www.roboti.us/index.html).
* Unzip the downloaded archive into `~/.mujoco/mujoco200` and place your license key file `mjkey.txt` at `~/.mujoco`.
* Use the env variables `MUJOCO_PY_MJKEY_PATH` and `MUJOCO_PY_MUJOCO_PATH` to specify the MuJoCo license key path and the MuJoCo directory path.
* Append the MuJoCo subdirectory bin path into the env variable `LD_LIBRARY_PATH`.

Install the following libraries:
```sh
sudo apt update
sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3
```

Install dependencies:
```sh
conda env create -f conda_env.yml
conda activate drqv2
```

Train the agent:
```sh
python train.py task=quadruped_walk
```

Monitor results:
```sh
tensorboard --logdir exp_local
```

## License
The majority of DrQ-v2 is licensed under the MIT license, however portions of the project are available under separate license terms: DeepMind is licensed under the Apache 2.0 license.
