from pathlib import Path
import yaml
import wandb


def clean_config(config: dict) -> dict:
    return {
        k: v['value'] if isinstance(v, dict) and 'value' in v else v
        for k, v in config.items()
    }


def get_wandb_info_from_run_dir(root_dir: str) -> dict:
    """
    From a known run root directory, find the corresponding wandb info.

    Args:
        root_dir: The root directory of the run.

    Returns:
        A dictionary containing the run_id, wandb run object, wandb_path,
        root_dir, config, and history dataframe.
    """

    api = wandb.Api()

    for wandb_path in Path('wandb').glob('run-*'):
        config_path = wandb_path / 'files/config.yaml'

        if not config_path.exists():
            continue

        with open(config_path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        if config.get('root_dir', dict(value=None))['value'] == str(root_dir) + '/':
            config = clean_config(config)
            run_id = wandb_path.name.split('-')[-1]
            run = api.run("drcope/mlrl/" + run_id)
            return {
                'run_id': run_id,
                'run': run,
                'wandb_path': wandb_path,
                'root_dir': root_dir,
                'config': config,
                'history': run.history()
            }

    raise ValueError('Could not find wandb info for model_dir: {}'.format(root_dir))
