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
    r'.*\.norm': 'my_layer',  # See 'Norm' in nets.py for variable creation.
}


class XlaVarsMirroredTest(tf.test.TestCase):

  def test_static_loop(self):
    # Google3: Works fine.
    # OSS TF 2.7: Fails (paste.googleplex.com/5440569951125504)
    helpers.time_train(repeats=10, kwargs={**KWARGS, r'.*\.unroll': True})

  def test_symbolic_loop(self):
    # Google3: Fails (paste.googleplex.com/5058329773604864)
    # OSS TF 2.7: Fails (paste.googleplex.com/5440569951125504)
    helpers.time_train(repeats=10, kwargs={**KWARGS, r'.*\.unroll': False})


if __name__ == '__main__':
  tf.test.main()
