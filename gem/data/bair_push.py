import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import torchvision
from multiprocessing import Pool
from google.protobuf.json_format import MessageToDict
import pickle, os, re, time
from functools import partial

from ..utils import save_npz
from .utils import get_unpack_functions
from .base import SequenceDataset
from .wrapper import ActionShift, SeparateImage, KeyMap, Split, ToTensor, multiple_wrappers

def load_bair_push(key='image_main', image_per_file=2):
    """
        key : choose from image_main, image_aux1
    """
    ROOT = 'dataset/bair_push/'
    TF_PATH = os.path.join(ROOT, 'tfrecords')
    BUILD_PATH = os.path.join(ROOT, 'build')

    if not os.path.exists(BUILD_PATH):
        if not os.path.exists(TF_PATH):
            raise FileExistsError('Please run dataset/bair_push/download.sh first!')
        else:
            print('Converting from tf_record ...')
            converter = BairConverter(TF_PATH, BUILD_PATH)
            converter.convert()

    config = {
        "c" : 3,
        "h" : 64,
        "w" : 64,
    }

    wrapper = multiple_wrappers([
        partial(KeyMap, key_pairs=[(key, 'image')]),
        partial(SeparateImage, image_per_file=image_per_file),
        ToTensor,
    ])

    trainset = wrapper(SequenceDataset(os.path.join(BUILD_PATH, 'train')))
    valset = wrapper(SequenceDataset(os.path.join(BUILD_PATH, 'val')))
    testset = wrapper(SequenceDataset(os.path.join(BUILD_PATH, 'test')))

    return (trainset, valset, testset, config)

def load_bair_push_seq(key="image_main", horizon=30, fix_start=True):
    ROOT = 'dataset/bair_push/'
    TF_PATH = os.path.join(ROOT, 'tfrecords')
    BUILD_PATH = os.path.join(ROOT, 'build')

    if not os.path.exists(BUILD_PATH):
        if not os.path.exists(TF_PATH):
            raise FileExistsError('Please run dataset/bair_push/download.sh first!')
        else:
            converter = BairConverter(TF_PATH, BUILD_PATH)
            converter.convert()

    config = {
        "obs" : (3, 64, 64),
        "action" : 4,
        'reward' : None,
    }

    wrapper = multiple_wrappers([
        # ActionShift,
        partial(KeyMap, key_pairs=[(key, 'image')]),
        partial(Split, horizon=horizon, fix_start=fix_start),
        ToTensor,
    ])

    trainset = wrapper(SequenceDataset(os.path.join(BUILD_PATH, 'train')))
    valset = wrapper(SequenceDataset(os.path.join(BUILD_PATH, 'val')))
    testset = wrapper(SequenceDataset(os.path.join(BUILD_PATH, 'test')))

    return (trainset, valset, testset, config)

class BairConverter(object):
    """
        Convert Bair Dataset from TFRecord to separate files
    """
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path
        self.datasets = self.get_sets()
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        self.sess = tf.Session()

    def get_sets(self):
        """
            Return the data folder inside the input path that we need to convert
        """
        return ('train', 'test', 'val')
    
    def check_config(self):
        """
            Check the configure in the TFrecode
        """
        # assume each file is the same
        folder = os.path.join(self.input_path, self.datasets[0])
        filename = os.path.join(folder, os.listdir(folder)[0])

        # find one sequence
        record = tf.python_io.tf_record_iterator(filename)
        traj = next(record)
        dict_message = MessageToDict(tf.train.Example.FromString(traj))
        feature = dict_message['features']['feature']
        
        # find the keys and frame numbers in the sequence
        num_set = set()
        key_set = set()
        for key in feature.keys():
            parse = re.findall(r'(\d+)(.*)', key)[0]
            num_set.add(int(parse[0]))
            key_set.add(parse[1])
        self.sequence_size = max(num_set) + 1
        self.keys = list(key_set)

        # find the data structure for each key
        self.structure = list()
        for key in self.keys:
            data = feature['0' + key]
            self.structure.append(list(data.keys())[0])
        self.functions = get_unpack_functions(self.structure)

        print('----------------------------------------------')
        print('Sequence size: {}'.format(self.sequence_size))
        for i in range(len(self.keys)):
            print(self.keys[i], self.structure[i])
        print('----------------------------------------------')

        # get image size
        for i in range(len(self.keys)):
            if 'image' in self.keys[i]:
                image_key = self.keys[i]
                image_function = self.functions[i]

        example = tf.train.Example()
        example.ParseFromString(traj)
        feature = example.features.feature
        image_raw = image_function(feature['16' + image_key])[0]
        image_flatten = self.sess.run(tf.decode_raw(image_raw, tf.uint8))

        image_size = int(np.sqrt(image_flatten.shape[0] // 3)) # assume images are square
        self.image_shape = (image_size, image_size, 3)
        image = image_flatten.reshape(self.image_shape)

        # save the config 
        config = {}
        for key, function in  zip(self.keys, self.functions):
            raw = function(feature['0' + key])
            if 'image' in key:
                image_flatten = self.sess.run(tf.decode_raw(raw[0], tf.uint8))
                data = image_flatten.reshape(self.image_shape)
            else:
                data = np.array(raw)
            config[key] = data.shape
        print('-' * 50)
        print(config)
        print('-' * 50)

        self.parser = partial(_parse, keys=self.keys, functions=self.functions, sequence_size=self.sequence_size, image_shape=self.image_shape)
    
    def convert(self):
        self.check_config()
        pool = Pool(os.cpu_count())
        for dataset in self.datasets:
            print('-' * 50)
            print('In dataset {}'.format(dataset))
            print('-' * 50)
            input_folder = os.path.join(self.input_path, dataset)
            output_folder = os.path.join(self.output_path, dataset)
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            filenames = sorted(os.listdir(input_folder))

            processes = []
            for filename in filenames:
                p = pool.apply_async(convert_single, (os.path.join(input_folder, filename), output_folder, self.parser))
                processes.append(p)

            for p in processes:
                p.get()

        pool.close()
        pool.join()

def _parse(traj, keys, functions, sequence_size, image_shape):
    example = tf.train.Example()
    example.ParseFromString(traj)
    feature = example.features.feature
    dict_data = dict()
    for key, func in zip(keys, functions):
        list_data = []
        for i in range(sequence_size):
            raw = func(feature[('%d' % i) + key])
            if 'image' in key:
                image_flatten = np.array([b for b in raw[0]], dtype=np.uint8)
                data = image_flatten.reshape(image_shape)
            else:
                data = np.array(raw)
            list_data.append(data[np.newaxis])
        dict_data[key.split('/')[1]] = np.concatenate(list_data, axis=0)
    return dict_data

def convert_single(input_traj, output_folder, parser):
    traj_name = input_traj.split('/')[-1]
    index = min(map(int, re.findall(r'(\d+)', traj_name)))
    record = tf.python_io.tf_record_iterator(input_traj)
    for traj in record:
        dict_data = parser(traj)
        traj_file = os.path.join(output_folder, f'traj_{index}_30.npz')
        save_npz(traj_file, dict_data)
        print('traj {} is finished!'.format(index)) 
        index += 1