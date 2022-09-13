import embodied


class Dispatch(embodied.Replay):

  def __init__(self, replays):
    self.replays = replays
    self.index = 0

  def __len__(self):
    return len(self.replays[0])

  @property
  def stats(self):
    return self.replays[0].stats

  def add(self, tran, worker=0):
    return self.replays[0].add(tran, worker)

  def add_traj(self, traj):
    return self.replays[0].add_traj(traj)

  def dataset(self):
    # Assuming that the agent creates separate dataset generators for each
    # entry in the training batch, each batch index will be taken from a
    # different replay buffer.
    dataset = self.replays[self.index % len(self.replays)].dataset()
    self.index += 1
    return dataset
