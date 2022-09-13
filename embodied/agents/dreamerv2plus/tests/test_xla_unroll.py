import os
import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).parent))
os.environ['TF_FUNCTION_JIT_COMPILE_DEFAULT'] = '1'

import tensorflow as tf

import helpers

SLACK = 1.3


class XLAUnrollTest(tf.test.TestCase):

  def test_train_static(self):
    kwargs = {r'.*\.unroll': True}
    times = helpers.time_train(repeats=5, kwargs=kwargs)
    assert times[0] <= SLACK * 520, times[0]
    assert min(times[1:]) <= SLACK * 0.094, times[1:]

  def test_train_dynamic(self):
    kwargs = {r'.*\.unroll': False}
    times = helpers.time_train(repeats=5, kwargs=kwargs)
    assert times[0] <= SLACK * 85, times[0]
    assert min(times[1:]) <= SLACK * 0.120, times[1:]

  def test_run_static(self):
    kwargs = {r'.*\.unroll': True}
    duration = helpers.time_run_small(kwargs)
    assert duration < SLACK * 190, duration

  def test_run_dynamic(self):
    kwargs = {r'.*\.unroll': False}
    duration = helpers.time_run_small(kwargs)
    assert duration < SLACK * 110, duration


if __name__ == '__main__':
  tf.test.main()
