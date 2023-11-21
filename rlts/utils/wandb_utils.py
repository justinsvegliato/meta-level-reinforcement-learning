import os
from pathlib import Path
import yaml
import wandb
from typing import Optional


WANDB_ENV_VAR = "WANDB_API_KEY"


def setup_wandb_api_key(api_key_file: str):
    """
    Set the WANDB_API_KEY environment variable to the value in the given file.

    Args:
        api_key_file: The path to the file containing the API key.
    """
    api_key_file = Path(api_key_file)
    if not api_key_file.exists():
        print(f'Could not find api key file: {api_key_file}. '
              f'Will prompt user to log-in.')
        return

    with open(api_key_file) as f:
        api_key = f.read().strip()

    os.environ[WANDB_ENV_VAR] = api_key


def clean_config(config: dict) -> dict:
    return {
        k: v['value'] if isinstance(v, dict) and 'value' in v else v
        for k, v in config.items()
    }


def get_wandb_config(wandb_dir) -> Optional[dict]:
    wandb_dir = Path(wandb_dir)
    config_path = wandb_dir / 'files/config.yaml'

    if not config_path.exists():
        return None

    with open(config_path) as f:
        return clean_config(yaml.load(f, Loader=yaml.FullLoader))


def get_wandb_info(wandb_dir: Path) -> dict:
    wandb_dir = Path(wandb_dir)

    config = get_wandb_config(wandb_dir)
    wandb_files = list(wandb_dir.glob('*.wandb'))
    if len(wandb_files) == 0:
        raise ValueError('Could not find wandb file in {}'.format(wandb_dir))

    run_id = wandb_files[0].stem.split('-')[1]
    try:
        api = wandb.Api()
        run = api.run("drcope/rlts/" + run_id)
        history = run.history()
    except Exception as e:
        print('Failed to get run info from WandB API: ', e)
        run = None
        history = None

    return {
        'run_id': run_id,
        'run': run,
        'wandb_path': wandb_dir,
        'config': config,
        'history': history,
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
    root_dir = Path(root_dir)

    if (root_dir / 'wandb').exists():
        wandb_path = root_dir / 'wandb/latest-run'
        info = get_wandb_info(wandb_path)
        info['root_dir'] = root_dir
        return info

    wandb_folder = Path('wandb')

    if not wandb_folder.exists():
        raise ValueError('Could not find wandb folder')

    for wandb_path in wandb_folder.glob('run-*'):
        config = get_wandb_config(wandb_path)

        if config is None:
            continue

        if config.get('root_dir') == str(root_dir) + '/':
            info = get_wandb_info(wandb_path)
            info['root_dir'] = root_dir
            return info

    raise ValueError('Could not find wandb info for model_dir: {}'.format(root_dir))
