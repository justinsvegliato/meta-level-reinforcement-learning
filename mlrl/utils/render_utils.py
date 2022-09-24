from IPython.display import HTML, clear_output
import base64

import io
import numpy as np
from tf_agents.environments import TFEnvironment
from tf_agents.policies import TFPolicy

import chess
import chess.svg

import cairosvg
from PIL import Image
from io import BytesIO
import imageio
import imageio.core.util


import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


def plot_to_array(fig: plt.Figure) -> np.ndarray:
    """
    Converts a matplotlib figure to a numpy array.
    """
    io_buf = io.BytesIO()
    fig.savefig(io_buf, format='raw')
    io_buf.seek(0)
    img_arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                         newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
    io_buf.close()
    return img_arr


def render_chess_board(board, shape=(500, 500)):
    try:
        lastmove = board.peek()
    except IndexError:
        lastmove = None

    svg = chess.svg.board(board, lastmove=lastmove)
    return svg_to_array(svg, shape=shape)


def svg_to_array(svg: str, shape=(128, 128)) -> np.ndarray:
    """
    Converts an SVG string to a numpy array.

    Args:
        svg (str): A string containing the SVG to convert.

    Returns:
        np.ndarray: A numpy array containing the rasterised image.
    """
    png = cairosvg.svg2png(bytestring=svg, 
                           output_width=shape[0], 
                           output_height=shape[1])
    img = Image.open(BytesIO(png))
    return np.array(img)


def embed_mp4(filename: str, clear_before=True) -> HTML:
    """Embeds an mp4 file in the notebook."""
    video = open(filename, 'rb').read()
    b64 = base64.b64encode(video)
    tag = '''
    <video width="640" height="480" controls>
    <source src="data:video/mp4;base64,{0}" type="video/mp4">
    Your browser does not support the video tag.
    </video>'''.format(b64.decode())
    if clear_before:
        clear_output()

    return HTML(tag)


def create_policy_eval_video(policy: TFPolicy,
                             env: TFEnvironment,
                             filename: str = 'video',
                             max_steps: int = 60,
                             fps: int = 1) -> str:
    """
    Creates and saves a video of the policy being evaluating in an environment.

    Args:
        policy (TFPolicy): The policy to evaluate.
        env (TFEnvironment): The environment to evaluate the policy in.
        filename (str): The name of the file to save the video to.
        max_steps (int): The maximum number of steps to run the policy for.
        fps (int): The frames per second of the video.

    Returns:
        str: The path to the saved video.
    """

    if not filename.endswith('.mp4'):
        filename = filename + '.mp4'

    env.reset()

    with imageio.get_writer(filename, fps=fps) as video:

        video.append_data(env.render().numpy()[0])

        for _ in range(max_steps):
            action_step = policy.action(env.current_time_step())
            env.step(action_step.action)

            video.append_data(env.render().numpy()[0])

    return filename
