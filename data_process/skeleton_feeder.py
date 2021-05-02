# sys
import torch
import numpy as np
import pickle
import random
# operation
from .utils import skeleton
from config import params
from glob import glob
import os

class SkeletonFeeder(torch.utils.data.Dataset):
    """ Feeder for skeleton-based action recognition
    Arguments:
        data_path: the path to '.npy' data, the shape of data should be (N, C, T, V, M)
        label_path: the path to label
        random_choose: If true, randomly choose a portion of the input sequence
        random_shift: If true, randomly pad zeros at the begining or end of sequence
        window_size: The length of the output sequence
        normalization: If true, normalize input sequence
        debug: If true, only use the first 100 samples
    """
    def __init__(self,
                 mode='train',
                 cliplen=30,
                 random_choose=False,
                 random_move=False,
                 window_size=-1,
                 debug=False,
                 mmap=True):
        self.debug = debug
        data_root = params['root']
        if mode == 'train':
            self.data_path = glob('/home/xmj/dataset/nturgbd/60train_data.npy') #to change to your own data path
            self.label_path = glob('/home/xmj/dataset/nturgbd/60train_label.pkl')  #to change to your own data path

        else:
            self.data_path = glob('/home/xmj/dataset/nturgbd/60val_data.npy')  #to change to your own data path
            self.label_path = glob('/home/xmj/dataset/nturgbd/60val_label.pkl')  #to change to your own data path

        self.random_choose = random_choose
        self.random_move = random_move
        self.window_size = window_size
        self.cliplen = cliplen

        self.load_data(mmap)

    def load_data(self, mmap):
        # data: N C V T M

        # load label
        for i, (data_path, label_path) in enumerate(zip(self.data_path, self.label_path)):
            with open(label_path, 'rb') as f:
                sample_name, label = pickle.load(f)

        # load data
            if mmap:
                data = np.load(data_path, mmap_mode='r')
            else:
                data = np.load(data_path)

            if self.debug:
                self.label = label[0:10]
                self.data = data[0:10]
                self.sample_name = sample_name[0:100]
                break

            if i == 0:
                self.data = data
                self.label = label
                self.sample_name = sample_name
            else:
                self.data = np.concatenate((self.data, data), 0)
                self.label = np.concatenate((self.label, label), 0)
                self.sample_name = self.sample_name.extend(sample_name)
                
        self.data = self.data[:, :, 0:100, :, :]

        self.N, self.C, self.T, self.V, self.M = self.data.shape
        
    def get_data(self):
        return self.data
        
        
        

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        # get data
        data_numpy = np.array(self.data[index])

        label = self.label[index]

        # processing
        if self.random_choose:
            data_numpy = skeleton.random_choose(data_numpy, self.window_size)
        elif self.window_size > 0:
            data_numpy = skeleton.auto_pading(data_numpy, self.window_size)
        if self.random_move:
            data_numpy = skeleton.random_move(data_numpy)

        return data_numpy, label
