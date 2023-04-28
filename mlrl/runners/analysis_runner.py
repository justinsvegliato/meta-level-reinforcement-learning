import json
import pandas as pd
from typing import Callable


class AnalysisRunner:

    def __init__(self, name: str,
                 output_dir: str = 'outputs/analysis',
                 **metadata):
        self.data = None
        self.name = name
        self.output_dir = output_dir
        self.metadata = {'name': name, **metadata}

    @property
    def results_path(self):
        return f'{self.output_dir}/{self.name}/results.csv'

    @property
    def metadata_path(self):
        return f'{self.output_dir}/{self.name}/metadata.json'

    def reload(self) -> pd.DataFrame:
        self.data = pd.read_csv(self.results_path)
        with open(self.metadata_path, 'r') as f:
            self.metadata = json.load(f)
        return self.data

    def run(self, analysis_fn: Callable) -> pd.DataFrame:
        self.data = analysis_fn()
        self.data.to_csv(self.results_path)
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f)
        return self.data
