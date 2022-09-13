# DayDreamer

## A1

```
rm -rf ~/logdir/run1
```

```
CUDA_VISIBLE_DEVICES=1 py embodied/agents/dreamerv2plus/train.py --configs a1 --task a1_sim --run learning --tf.platform gpu --logdir ~/logdir/run1
```

```
CUDA_VISIBLE_DEVICES=2 py embodied/agents/dreamerv2plus/train.py --configs a1 --task a1_real --run acting --tf.platform gpu --env.kbreset True --imag_horizon 1 --replay_chunk 8 --replay_fixed.minlen 32 --imag_horizon 1 --logdir ~/logdir/run1
```

## XArm

```
rm -rf ~/logdir/run1
```

```
CUDA_VISIBLE_DEVICES=0 py embodied/agents/dreamerv2plus/train.py --configs xarm --run learning --task xarm_dummy --tf.platform gpu --logdir ~/logdir/run1
```

```
CUDA_VISIBLE_DEVICES=-1 py embodied/agents/dreamerv2plus/train.py --configs xarm --run acting --task xarm_real --env.kbreset True --tf.platform cpu --tf.jit False --logdir ~/logdir/run1
```

## UR

```
rm -rf ~/logdir/run1
```

```
CUDA_VISIBLE_DEVICES=1 python embodied/agents/dreamerv2plus/train.py --configs ur5 --run learning --task ur5_dummy --tf.platform gpu --logdir ~/logdir/run11
```

```
CUDA_VISIBLE_DEVICES=0 python embodied/agents/dreamerv2plus/train.py --configs ur5 --run acting --task ur5_real --env.kbreset True --tf.platform cpu --tf.jit False --logdir ~/logdir/run11
```
