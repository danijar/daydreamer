import os
import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).parent))

# Enable jit_compile=True for all tf.functions.
os.environ['TF_FUNCTION_JIT_COMPILE_DEFAULT'] = '1'

import tensorflow as tf

import helpers

SLACK = 1.3

KWARGS = {
    'tf.platform': 'multi_gpu',
    'env.parallel': 'thread',
    'replay_fixed.length': 8,
}

KWARGS_FAST = {
    'tf.platform': 'multi_gpu',
    'env.parallel': 'thread',
    'replay_fixed.length': 8,
    'batch_size': 8,
    'rssm.deter': 128,
    'rssm.units': 128,
    'rssm.stoch': 8,
    r'.*\.cnn_depth$': 16,
}


class MirroredXLATest(tf.test.TestCase):

  # def test_train(self):
  #   os.environ['TF_FUNCTION_JIT_COMPILE_DEFAULT'] = '1'
  #   gpus = tf.config.experimental.list_physical_devices('GPU')
  #   assert len(gpus) == 2, gpus
  #   times = helpers.time_train(repeats=10, kwargs=KWARGS)
  #   assert times[0] <= SLACK * 185, times[0]
  #   assert min(times[1:]) <= SLACK * 0.600, times[1:]

  # def test_train_fast(self):
  #   gpus = tf.config.experimental.list_physical_devices('GPU')
  #   assert len(gpus) == 2, gpus
  #   times = helpers.time_train(repeats=10, kwargs=KWARGS_FAST)
  #   assert times[0] <= SLACK * 185, times[0]
  #   assert min(times[1:]) <= SLACK * 0.600, times[1:]

  # def test_policy(self):
  #   os.environ['TF_FUNCTION_JIT_COMPILE_DEFAULT'] = '1'
  #   gpus = tf.config.experimental.list_physical_devices('GPU')
  #   assert len(gpus) == 2, gpus
  #   times = helpers.time_policy(repeats=10, kwargs=KWARGS)
  #   assert times[0] <= SLACK * 6, times[0]
  #   assert min(times[1:]) <= SLACK * 0.004, times[1:]

  # def test_report(self):
  #   os.environ['TF_FUNCTION_JIT_COMPILE_DEFAULT'] = '1'
  #   gpus = tf.config.experimental.list_physical_devices('GPU')
  #   assert len(gpus) == 2, gpus
  #   times = helpers.time_report(repeats=10, kwargs=KWARGS)
  #   assert times[0] <= SLACK * 45, times[0]
  #   assert min(times[1:]) <= SLACK * 0.203, times[1:]

  # def test_run(self):
  #   os.environ['TF_FUNCTION_JIT_COMPILE_DEFAULT'] = '1'
  #   gpus = tf.config.experimental.list_physical_devices('GPU')
  #   assert len(gpus) == 2, gpus
  #   duration = helpers.time_run_small(KWARGS_FAST)
  #   assert duration < SLACK * 140, duration

  def test_train(self):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    assert len(gpus) == 2, gpus
    helpers.time_train(repeats=10, kwargs=KWARGS_FAST)

  def test_run(self):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    assert len(gpus) == 2, gpus
    helpers.time_run_small(KWARGS_FAST)


if __name__ == '__main__':
  tf.test.main()
