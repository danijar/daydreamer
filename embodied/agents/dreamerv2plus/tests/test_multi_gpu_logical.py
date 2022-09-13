import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).parent))

import tensorflow as tf

import helpers

SLACK = 1.3

KWARGS = {
    'tf.platform': 'multi_gpu',
    'tf.logical_gpus': 2,
    'replay_fixed.length': 8,
}

# KWARGS_FAST = {
#     'tf.platform': 'multi_gpu',
#     'tf.logical_gpus': 2,
#     'replay_fixed.length': 8,
#     'batch_size': 8,
#     'rssm.deter': 128,
#     'rssm.units': 128,
#     'rssm.stoch': 8,
#     r'.*\.cnn_depth$': 16,
# }


class MultiGPULogicalTest(tf.test.TestCase):

  def test_train(self):
    times = helpers.time_train(repeats=10, kwargs=KWARGS)
    assert times[0] <= SLACK * 185, times[0]
    assert min(times[1:]) <= SLACK * 0.600, times[1:]

  # def test_train_fast(self):
  #   times = helpers.time_train(repeats=10, kwargs=KWARGS_FAST)
  #   assert times[0] <= SLACK * 185, times[0]
  #   assert min(times[1:]) <= SLACK * 0.600, times[1:]

  def test_policy(self):
    times = helpers.time_policy(repeats=10, kwargs=KWARGS)
    assert times[0] <= SLACK * 6, times[0]
    assert min(times[1:]) <= SLACK * 0.004, times[1:]

  def test_report(self):
    times = helpers.time_report(repeats=10, kwargs=KWARGS)
    assert times[0] <= SLACK * 45, times[0]
    assert min(times[1:]) <= SLACK * 0.203, times[1:]

  def test_run(self):
    duration = helpers.time_run_small(KWARGS)
    assert duration < SLACK * 140, duration


if __name__ == '__main__':
  tf.test.main()
