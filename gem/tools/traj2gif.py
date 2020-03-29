"""
    Save one traj in the form of gif

    Usage: 
        python -m gem.tools.traj2gif --traj <traj_file> --gif <gif_file> --fps <fps>

    Args:

        traj_file : str, the traj file you want to convert
        gif_file : str, the output gif
        fps : int, the fps for the saved gif, default : 30
"""

import os
import argparse

from ..utils import load_npz, create_dir, save_gif

INPUT_DIR = 'dataset/'
OUTPUT_DIR = 'outputs/gif'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--traj', type=str)
    parser.add_argument('--gif', type=str)
    parser.add_argument('--key', type=str, default='image')
    parser.add_argument('--fps', type=int, default=30)
    args = parser.parse_args()
    
    create_dir(OUTPUT_DIR)

    input_file = os.path.join(INPUT_DIR, args.traj)
    output_file = os.path.join(OUTPUT_DIR, args.gif)

    video = load_npz(input_file)[args.key]

    save_gif(output_file, video, args.fps)