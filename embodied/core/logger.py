import concurrent.futures
import json
import re
import time

import numpy as np

from . import path


class Logger:

  def __init__(self, step, outputs, multiplier=1):
    self._step = step
    self._outputs = outputs
    self._multiplier = multiplier
    self._last_step = None
    self._last_time = None
    self._metrics = []

  @property
  def step(self):
    return self._step

  def add(self, mapping, prefix=None):
    step = int(self._step) * self._multiplier
    for name, value in dict(mapping).items():
      name = f'{prefix}/{name}' if prefix else name
      value = np.array(value)
      if len(value.shape) not in (0, 2, 3, 4):
        raise ValueError(
            f"Shape {value.shape} for name '{name}' cannot be "
            "interpreted as scalar, image, or video.")
      self._metrics.append((step, name, value))

  def scalar(self, name, value):
    self.add({name: value})

  def image(self, name, value):
    self.add({name: value})

  def video(self, name, value):
    self.add({name: value})

  def write(self, fps=False):
    if fps:
      value = self._compute_fps()
      if value is not None:
        self.scalar('fps', value)
    if not self._metrics:
      return
    for output in self._outputs:
      output(tuple(self._metrics))
    self._metrics.clear()

  def _compute_fps(self):
    step = int(self._step) * self._multiplier
    if self._last_step is None:
      self._last_time = time.time()
      self._last_step = step
      return None
    steps = step - self._last_step
    duration = time.time() - self._last_time
    self._last_time += duration
    self._last_step = step
    return steps / duration


class AsyncOutput:

  def __init__(self, callback, parallel=True):
    self._callback = callback
    self._parallel = parallel
    if parallel:
      self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
      self._future = None

  def __call__(self, summaries):
    if self._parallel:
      if self._future:
        self._future.result()
      self._future = self._executor.submit(self._callback, summaries)
    else:
      self._callback(summaries)


class TerminalOutput:

  def __init__(self, pattern=r'.*'):
    self._pattern = re.compile(pattern)
    try:
      import rich.console
      self._console = rich.console.Console()
    except ImportError:
      self._console = None

  def __call__(self, summaries):
    step = max([s for s, _, _, in summaries])
    scalars = {k: float(v) for _, k, v in summaries if len(v.shape) == 0}
    scalars = {k: v for k, v in scalars.items() if self._pattern.search(k)}
    formatted = {k: self._format_value(v) for k, v in scalars.items()}
    if self._console:
      self._console.rule(f'[green bold]Step {step}')
      self._console.print(' [blue]/[/blue] '.join(
          f'{k} {v}' for k, v in formatted.items()))
      print('')
    else:
      message = ' / '.join(f'{k} {v}' for k, v in formatted.items())
      print(f'[{step}]', message, flush=True)

  def _format_value(self, value):
    if value == 0:
      return '0'
    elif 0.01 < abs(value) < 10000:
      value = f'{value:.2f}'
      value = value.rstrip('0')
      value = value.rstrip('0')
      value = value.rstrip('.')
      return value
    else:
      value = f'{value:.1e}'
      value = value.replace('.0e', 'e')
      value = value.replace('+0', '')
      value = value.replace('+', '')
      value = value.replace('-0', '-')
    return value


class JSONLOutput(AsyncOutput):

  def __init__(
      self, logdir, filename='metrics.jsonl', pattern=r'.*', parallel=True):
    super().__init__(self._write, parallel)
    self._filename = filename
    self._pattern = re.compile(pattern)
    self._logdir = path.Path(logdir)
    self._logdir.mkdirs()

  def _write(self, summaries):
    scalars = {k: float(v) for _, k, v in summaries if len(v.shape) == 0}
    scalars = {k: v for k, v in scalars.items() if self._pattern.search(k)}
    step = max(s for s, _, _, in summaries)
    with (self._logdir / self._filename).open('a') as f:
      f.write(json.dumps({'step': step, **scalars}) + '\n')


class TensorBoardOutput(AsyncOutput):

  def __init__(self, logdir, fps=20, parallel=True):
    super().__init__(self._write, parallel)
    self._logdir = str(logdir)
    if self._logdir.startswith('/gcs/'):
      self._logdir = self._logdir.replace('/gcs/', 'gs://')
    self._fps = fps
    self._writer = None

  def _write(self, summaries):
    import tensorflow as tf
    if not self._writer:
      self._writer = tf.summary.create_file_writer(
          self._logdir, max_queue=1000)
    self._writer.set_as_default()
    for step, name, value in summaries:
      if len(value.shape) == 0:
        tf.summary.scalar(name, value, step)
      elif len(value.shape) == 2:
        tf.summary.image(name, value, step)
      elif len(value.shape) == 3:
        tf.summary.image(name, value, step)
      elif len(value.shape) == 4:
        self._video_summary(name, value, step)
    self._writer.flush()

  def _video_summary(self, name, video, step):
    import tensorflow as tf
    import tensorflow.compat.v1 as tf1
    name = name if isinstance(name, str) else name.decode('utf-8')
    if np.issubdtype(video.dtype, np.floating):
      video = np.clip(255 * video, 0, 255).astype(np.uint8)
    try:
      T, H, W, C = video.shape
      summary = tf1.Summary()
      image = tf1.Summary.Image(height=H, width=W, colorspace=C)
      image.encoded_image_string = _encode_gif(video, self._fps)
      summary.value.add(tag=name, image=image)
      tf.summary.experimental.write_raw_pb(summary.SerializeToString(), step)
    except (IOError, OSError) as e:
      print('GIF summaries require ffmpeg in $PATH.', e)
      tf.summary.image(name, video, step)


class MlflowOutput:

  def __init__(self, run_name=None, resume_id=None, params={}, prefix=''):
    import os
    run_name = run_name or os.environ.get('MLFLOW_RUN_NAME')
    resume_id = resume_id or os.environ.get('MLFLOW_RESUME_ID')
    self._start_or_resume(run_name, resume_id)
    if params:
      self._log_params(params)
    self.prefix = prefix

  def __call__(self, summaries):
    import mlflow
    import datetime
    scalars = {k: float(v) for _, k, v in summaries if len(v.shape) == 0}
    step = max(s for s, _, _, in summaries)
    scalars['step'] = step
    scalars['timestamp'] = datetime.datetime.now().timestamp()
    if self.prefix:
      scalars = {f'{self.prefix}{k}': v for k, v in scalars.items()}
    mlflow.log_metrics(scalars, step=step)

  def _start_or_resume(self, run_name, resume_id=None):
    import os
    import mlflow
    resume_run_id = None
    print('Mlflow tracking uri:', os.environ.get('MLFLOW_TRACKING_URI', 'local'))
    if resume_id:
        runs = mlflow.search_runs(filter_string=f'tags.resume_id="{resume_id}"')
        if len(runs) > 0:
            resume_run_id = runs['run_id'].iloc[0]
    if resume_run_id:
      run = mlflow.start_run(run_name=run_name, run_id=resume_run_id)
      print(f'Resumed mlflow run {run.info.run_id} ({resume_id}) in experiment {run.info.experiment_id}')
    else:
      run = mlflow.start_run(run_name=run_name, tags={'resume_id': resume_id or ''})
      print(f'Started mlflow run {run.info.run_id} ({resume_id}) in experiment {run.info.experiment_id}')
    return run

  def _log_params(self, params):
    import mlflow
    kvs = list(params.items())
    for i in range(0, len(kvs), 100):  # log_params() allows max 100
      try:
        mlflow.log_params(dict(kvs[i:i+100]))
      except Exception as ex:
        # This is normal when resuming run, mlflow complains when trying to change value of params
        print(f'WARN: error logging parameters ({ex})')


def _encode_gif(frames, fps):
  from subprocess import Popen, PIPE
  h, w, c = frames[0].shape
  pxfmt = {1: 'gray', 3: 'rgb24'}[c]
  cmd = ' '.join([
      'ffmpeg -y -f rawvideo -vcodec rawvideo',
      f'-r {fps:.02f} -s {w}x{h} -pix_fmt {pxfmt} -i - -filter_complex',
      '[0:v]split[x][z];[z]palettegen[y];[x]fifo[x];[x][y]paletteuse',
      f'-r {fps:.02f} -f gif -'])
  proc = Popen(cmd.split(' '), stdin=PIPE, stdout=PIPE, stderr=PIPE)
  for image in frames:
    proc.stdin.write(image.tobytes())
  out, err = proc.communicate()
  if proc.returncode:
    raise IOError('\n'.join([' '.join(cmd), err.decode('utf8')]))
  del proc
  return out
