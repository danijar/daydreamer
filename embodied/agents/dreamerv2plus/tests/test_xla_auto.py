import os
import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).parent))
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2'

import tensorflow as tf

import helpers

SLACK = 1.3


class XLAAutoTest(tf.test.TestCase):

  def test_train(self):
    times = helpers.time_train(repeats=5)
    assert times[0] <= SLACK * 318, times[0]
    assert min(times[1:]) <= SLACK * 0.115, times[1:]

  def test_policy(self):
    times = helpers.time_policy(repeats=5)
    assert times[0] <= SLACK * 8, times[0]
    assert min(times[1:]) <= SLACK * 0.004, times[1:]

  def test_report(self):
    times = helpers.time_report(repeats=5)
    assert times[0] <= SLACK * 107, times[0]
    assert min(times[1:]) <= SLACK * 0.070, times[1:]

  def test_run(self):
    duration = helpers.time_run_small()
    assert duration < SLACK * 133, duration


if __name__ == '__main__':
  tf.test.main()
