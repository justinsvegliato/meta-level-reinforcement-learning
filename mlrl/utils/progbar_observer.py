from typing import List
from tensorflow.keras.utils import Progbar
import numpy as np

from tf_agents.trajectories.trajectory import Trajectory
from tf_agents.metrics.py_metric import PyMetric


class ProgressBarObserver:

    def __init__(self,
                 max_progress: int,
                 update_interval: int = 5,
                 metrics: List[PyMetric] = None):
        self.max_progress: int = max_progress
        self.update_interval: int = update_interval
        self.progress: int = 0
        self.bar = Progbar(max_progress)
        self.metrics = metrics or []

    def __call__(self, traj: Trajectory):
        self.progress += np.sum(~traj.is_boundary())

        if self.progress % self.update_interval == 0:
            values = [(metric.name, metric.result()) for metric in self.metrics]
            self.bar.update(self.progress, values=values)

        if self.progress >= self.max_progress:
            self.reset()

    def reset(self):
        self.progress = 0
        self.bar = Progbar(self.max_progress)
