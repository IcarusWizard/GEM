"""
    Get single image from traj

    Usage: 
        python -m gem.tools.get_image --traj <traj_file> --img <image_file> --index <index>

    Args:

        traj_file : str, the traj file you want to convert
        image_file : str, the output image file
        index : int, index of the image in traj, default : 0
"""

import os
import argparse

import matplotlib.pyplot as plt

from ..utils import load_npz, create_dir

INPUT_DIR = 'dataset/'
OUTPUT_DIR = 'outputs/image'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--traj', type=str)
    parser.add_argument('--img', type=str)
    parser.add_argument('--key', type=str, default='image')
    parser.add_argument('--index', type=int, default=0)
    args = parser.parse_args()
    
    create_dir(OUTPUT_DIR)

    input_file = os.path.join(INPUT_DIR, args.traj)
    output_file = os.path.join(OUTPUT_DIR, args.img)

    image = load_npz(input_file)[args.key][args.index]

    plt.imsave(output_file, image)