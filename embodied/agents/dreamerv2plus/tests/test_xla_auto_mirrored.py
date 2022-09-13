import os
import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).parent))

os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2'

import tensorflow as tf

import helpers

SLACK = 1.3

KWARGS = {
    'tf.platform': 'multi_gpu',
    'tf.logical_gpus': 2,
    'replay.length': 8,
}


class XlaAutoMirroredTest(tf.test.TestCase):

  def test_static_loop(self):
    # Google3: Works fine.
    # OSS TF 2.7: Works fine.
    helpers.time_train(repeats=10, kwargs={**KWARGS, r'.*\.unroll': True})

  def test_symbolic_loop(self):
    # Google3: Works fine.
    # OSS TF 2.7: Works fine.
    helpers.time_train(repeats=10, kwargs={**KWARGS, r'.*\.unroll': False})


if __name__ == '__main__':
  tf.test.main()
