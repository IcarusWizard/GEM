import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Pool
from google.protobuf.json_format import MessageToDict
import pickle, os, re, time

def unpack_float_list(feature):
    return feature.float_list.value

def unpack_bytes_list(feature):
    return feature.bytes_list.value

def unpack_int64_list(feature):
    return feature.int64_list.value

def get_unpack_functions(structures):
    functions = []
    for structure in structures:
        if structure == "floatList":
            functions.append(unpack_float_list)
        elif structure == "bytesList":
            functions.append(unpack_bytes_list)
        elif structure == 'Int64List':
            functions.append(unpack_int64_list)
        else:
            raise Exception('no such type {}'.format(structure))
    return functions

def check_keys(name, keys):
    """
        check if name matches any keys

        Inputs:

            name : str
            keys : list[str]
    """
    for key in keys:
        if key in name:
            return True
    return False