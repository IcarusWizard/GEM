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
import torch
from tqdm import tqdm

from gem.models.sensor.run_utils import get_sensor_by_checkpoint
from gem.utils import load_npz, create_dir

MODEL_DIR = 'checkpoint/sensor'
IMAGE_DIR = 'outputs/image'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--image', type=str)
    parser.add_argument('--iter', type=int, default=5000)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    input_file = os.path.join(IMAGE_DIR, args.image)
    image = plt.imread(input_file)

    checkpoint = torch.load(os.path.join(MODEL_DIR, args.checkpoint + '.pt'), map_location='cpu')
    model = get_sensor_by_checkpoint(checkpoint).to(device)
    model.requires_grad_(False)

    image = torch.as_tensor((image / 255.0 - 0.5), dtype=torch.float32).permute(2, 0, 1).unsqueeze(dim=0).to(device)

    z = model.encode(image)
    _image = model.decode(z)[0].cpu().permute(1, 2, 0).numpy()

    _z = torch.zeros_like(z).requires_grad_()
    optim = torch.optim.Adam([_z], lr=1e-3)

    for i in tqdm(range(5000)):
        __image = model.decode(_z)

        loss = torch.sum((image - __image) ** 2)

        optim.zero_grad()
        loss.backward()
        optim.step()
    
    image = image[0].cpu().permute(1, 2, 0).numpy()
    __image = __image = model.decode(_z)[0].detach().cpu().permute(1, 2, 0).numpy()

    fig = plt.figure()
    plt.imshow(image)
    plt.title('orginal')

    fig = plt.figure()
    plt.imshow(_image)
    plt.title('reconstruction')

    fig = plt.figure()
    plt.imshow(__image)
    plt.title('find in latent')    

    plt.show()