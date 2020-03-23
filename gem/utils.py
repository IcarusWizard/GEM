import pickle
import os

def pickle_data(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def load_pkl(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def create_dir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)