
from tensorflow.keras.utils import Progbar


class ProgressBarObserver:

    def __init__(self, collect_sequence_length: int, update_interval: int = 128):
        self.collect_sequence_length: int = collect_sequence_length
        self.update_interval: int = update_interval
        self.progress: int = 0
        self.bar = Progbar(collect_sequence_length)

    def __call__(self, _):
        if self.progress >= self.collect_sequence_length:
            self.progress = 0

        self.progress += 1
        if self.progress % self.update_interval == 0:
            self.bar.update(self.progress)
