from rlts.meta.meta_env import MetaEnv
from rlts.utils.render_utils import save_video

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()


class ResultsAccumulator:

    def __init__(self, output_dir: Path = None):
        self.results = []
        self.episode_stats = []
        self.results_df = None
        self.episode_stats_df = None
        self.output_dir = Path(output_dir) if output_dir else None
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def __call__(self, results: dict):
        self.results.append(results)
        self.results_df = pd.DataFrame(self.results)
        if self.output_dir is not None:
            self.results_df.to_csv(self.output_dir / 'results.csv', index=False)

        self.episode_stats_df = pd.DataFrame(self.episode_stats)
        if self.output_dir is not None:
            self.episode_stats_df.to_csv(self.output_dir / 'episode_stats.csv', index=False)

        self.plot_results()

    def add_episode_stats(self, run_id: str, policy: str, percentile: float, stats: dict):
        self.episode_stats.append({
            'Run ID': run_id or '',
            'Meta-level Policy': policy,
            'Pretrained Percentile': percentile,
            'Number of Steps': stats['steps'],
            'Return': stats['return'],
            **stats
        })
        self.episode_stats_df = pd.DataFrame(self.episode_stats)
        if self.output_dir is not None:
            self.episode_stats_df.to_csv(self.output_dir / 'episode_stats.csv', index=False)

        self.plot_results()

    def plot_results(self):
        try:
            self.episode_stats_df = pd.DataFrame(self.episode_stats)
            self.results_df = pd.DataFrame(self.results)
            self.plot_rewritten_returns()
            self.plot_object_level_returns()

        except Exception:
            pass

    def plot_rewritten_returns(self):
        plot_name, plot_key = 'Mean Rewritten Meta Return', 'EvalRewrittenAverageReturn'

        plt.figure(figsize=(15, 10))

        sns.lineplot(data=self.results_df, x='pretrained_percentile', y=plot_key, hue='Meta-level Policy', alpha=0.25)
        sns.scatterplot(data=self.results_df, x='pretrained_percentile', y=plot_key, hue='Meta-level Policy', legend=False)

        plt.xlabel('Pretrained Percentile')
        plt.ylabel(plot_name)
        plt.title(f'{plot_name} vs Pretrained Percentile')

        if self.output_dir is not None:
            plt.savefig(self.output_dir / 'mean-object-return-vs-percentile.png')
        plt.close()

    def plot_object_level_returns(self):
        plot_name, plot_key = 'Mean Object-level Return', 'ObjectLevelMeanReward'

        plt.figure(figsize=(15, 10))
        sns.lineplot(data=self.episode_stats_df, x='Pretrained Percentile', y='Return', hue='Meta-level Policy', alpha=0.5)
        sns.scatterplot(data=self.results_df, x='pretrained_percentile', y=plot_key, hue='Meta-level Policy', legend=False)

        plt.xlabel('Pretrained Percentile')
        plt.ylabel(plot_name)
        plt.title(f'{plot_name} vs Pretrained Percentile')

        if self.output_dir is not None:
            plt.savefig(self.output_dir / 'object-return-vs-percentile.png')
        plt.close()


class MetaPolicyObjectlevelVideoMaker:

    def __init__(self,
                 output_dir: Path,
                 meta_env: MetaEnv,
                 video_name: str = 'object-level-video',
                 fps: int = 2):
        self.output_dir = output_dir
        self.video_name = video_name
        self.meta_env = meta_env
        self.fps = fps
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self.videos_created = 0
        self.frames = []

    def __call__(self, *_):
        self.frames.append(self.meta_env.render())
    
    def save_video(self, *_):
        if self.frames:
            path = str(self.output_dir / f'{self.video_name}-{self.videos_created}')
            save_video(self.frames, path, fps=self.fps)
            self.videos_created += 1
            self.frames = []
