import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).parent))

import tensorflow as tf

import helpers

SLACK = 1.3


class XLAOffTest(tf.test.TestCase):

  def test_train(self):
    times = helpers.time_train(repeats=5)
    assert times[0] <= SLACK * 75, times[0]
    assert min(times[1:]) <= SLACK * 0.351, times[1:]

  def test_policy(self):
    times = helpers.time_policy(repeats=5)
    assert times[0] <= SLACK * 5, times[0]
    assert min(times[1:]) <= SLACK * 0.004, times[1:]

  def test_report(self):
    times = helpers.time_report(repeats=5)
    assert times[0] <= SLACK * 45, times[0]
    assert min(times[1:]) <= SLACK * 0.203, times[1:]

  def test_run(self):
    duration = helpers.time_run_small()
    assert duration < SLACK * 47, duration


if __name__ == '__main__':
  tf.test.main()
