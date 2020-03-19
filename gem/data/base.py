import numpy as np
import os
import PIL
from queue import Queue

import torch, torchvision
from .utils import check_keys

# class VideoDataset(torch.utils.data.Dataset):
#     """
#         Return dataset contain video images
#         Keys and values:
#             observations : tensor B x T x C x H x W
#             actions : tensor B x T x A
#     """
#     def __init__(self, 
#                 path,
#                 dataset,
#                 horizon,
#                 fix_start=False):
#         super().__init__()
#         self.horizon = horizon
#         self.path = path
#         self.dataset = dataset
#         self.fix_start = fix_start
#         self.config = self.load_pkl(os.path.join(path, 'config.pkl'))
        
#         # use only one view
#         self.keys = []
#         self.shapes = []
#         for key, shape in self.config.items():
#             if 'image' in key:
#                 if 'image_main' in key or 'image_view0' in key:
#                     self.image_key = key[1:].split('/')[0]
#                     self.image_shape = shape
#                     self.keys.append(self.image_key)
#                     self.shapes.append(self.image_shape)
#             elif 'action' in key:
#                 self.action_key = key[1:]
#                 self.action_shape = shape
#                 self.keys.append(self.action_key)
#                 self.shapes.append(self.action_shape)
#         self.config = {'observations' : self.image_shape, 
#                        'actions' : self.action_shape}

#         # load trajlist
#         foldernames = sorted(os.listdir(os.path.join(path, dataset)))
#         self.trajlist = [os.path.join(path, dataset, foldername) for foldername in foldernames]

#         # find sequence length
#         actions = np.loadtxt(os.path.join(self.trajlist[0], "{}.txt".format(self.action_key)))
#         self.sequence_length = actions.shape[0]

#         assert self.horizon <= self.sequence_length, "horizon must smaller than sequence length, i.e. {} <= {}".format(
#             self.horizon,
#             self.sequence_length
#         )
        
#     def load_pkl(self, filename):
#         with open(filename, 'rb') as f:
#             data = pickle.load(f)
#         return data

#     def set_config(self, config):
#         self.config = config

#     def get_config(self):
#         return self.config
        
#     def __len__(self):
#         return len(self.trajlist)

#     def __getitem__(self, index):
#         # load data
#         traj_folder = self.trajlist[index]
#         # data = self.load_pkl(self.filelist[index])

#         # set start point
#         start = 0 if self.fix_start else random.randint(0, self.sequence_length - self.horizon)

#         # load data
#         output = {}

#         imgs = []
#         for i in range(start, start + self.horizon):
#             img = plt.imread(os.path.join(traj_folder, "{}_{}.jpg".format(self.image_key, i)))
#             imgs.append(img[np.newaxis])
#         imgs = np.concatenate(imgs, axis=0)
#         imgs = imgs / 255.0
#         imgs = np.transpose(imgs, (0, 3, 1, 2))
#         output['observations'] = torch.tensor(imgs, dtype=torch.float32)

#         actions = np.loadtxt(os.path.join(traj_folder, "{}.txt".format(self.action_key)))  
#         actions = actions[start : start + self.horizon]
#         output['actions'] = torch.tensor(actions, dtype=torch.float32)
        
#         return output

class ImageDataset(torch.utils.data.Dataset):
    """
        Base image dataset, load everything matches keys in path

        Inputs:

            path : str, path to the dataset
            keys : list[str]
            transform : func[PIL.Image -> tensor]
    """
    def __init__(self, path, keys=['png', 'jpg'], transform=None):
        self.path = path
        self.keys = keys
        self.transform = transform
        
        self.file_list = []

        # search for any matched image in path and its subfolders
        q = Queue()
        q.put(path)
        while not q.empty():
            folder = q.get()
            for file_name in os.listdir(folder):
                if os.path.isdir(file_name):
                    q.put(os.path.join(folder, file_name)) # add subfolder
                else:
                    if check_keys(file_name, keys):
                        self.file_list.append(os.path.join(folder, file_name))

    def __getitem__(self, index):
        image = PIL.Image.open(self.file_list[index])

        if self.transform is not None:
            image = self.transform(image)

        return image,

    def __len__(self):
        return len(self.file_list)

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--dataset', type=str, default='test')
#     parser.add_argument('--horizon', type=int, default=10)

#     args = parser.parse_args()

#     dataset = VideoDataset('data/bair', args.dataset, horizon=args.horizon, fix_start=True)
#     config = dataset.get_config()
#     print(config)
#     loader = torch.utils.data.DataLoader(dataset, batch_size=16, num_workers=4, shuffle=False)
#     count = 0
#     start = time.time()

#     gif_path = os.path.join('gt', '{}_{}'.format(args.dataset, args.horizon))
#     if not os.path.exists(gif_path):
#         os.makedirs(gif_path)

#     for data in loader:
#         end = time.time()
#         print(end - start)
#         start = end
#         imgs = data['observations'][0]
#         torch_save_gif(os.path.join(gif_path, '{}.gif'.format(count)), imgs, fps=10)
#         count += 1