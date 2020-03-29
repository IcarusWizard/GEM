"""
    Break up a gif file to separate images

    Usage: 
        python -m gem.tools.gif_breakup --gif <gif_file>

    Args:

        gif_file : str, the input gif
"""

import os
import argparse

from ..utils import create_dir
from imageio import mimread
import matplotlib.pyplot as plt

INPUT_DIR = 'outputs/gif'
OUTPUT_DIR = 'outputs/breakup'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gif', type=str)
    args = parser.parse_args()

    name = args.gif.split('.')[0]

    input_file = os.path.join(INPUT_DIR, args.gif)
    output_folder = os.path.join(OUTPUT_DIR, name)
    create_dir(output_folder)

    video = mimread(input_file)

    for i, image in enumerate(video):
        plt.imsave(os.path.join(output_folder, f'{i}.jpg'), image)