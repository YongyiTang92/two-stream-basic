import torch
import torch.nn as nn
import os
import torchvision
import subprocess
import math
import shutil
import random
from torch.autograd import Variable
from PIL import Image
from utilities import save_checkpoint, read_checkpointm
from logger import Logger
from model import Seq2SeqModel
import data_utils
import time
import h5py
import sys
from video_data_loader import ucf101_rgb_loader_basic_train, ucf101_rgb_loader_basic_test
from model import rgb_resnet18_basic


class solver(object):
    def __init__(self, FLAGS, train_dir, summaries_dir):
        self.FLAGS = FLAGS
        self.summaries_dir = summaries_dir
        self.logger = Logger(self.summaries_dir)
        self.train_dir = train_dir
        self.model = self.create_model()
        self.creat_data_loader()

    def create_model(self, actions, sampling=False):
        """Create translation model and initialize or load parameters in session."""

        model = rgb_resnet18_basic(self.FLAGS)

        if self.FLAGS.load <= 0:
            print("Creating model with fresh parameters.")
            self.start_step = 0
        else:
            print(self.train_dir)
            model_exist, checkpoint = read_checkpoint(self.train_dir)
            if model_exist:
                model.model.load_state_dict(checkpoint['state_dict'])
                self.start_step = checkpoint['step'] + 1

        return model

    def creat_data_loader(self):
        train_dataset = ucf101_rgb_loader_basic_train(self.FLAGS.data, self.FLAGS.rgb_file)
        self.test_dataset = ucf101_rgb_loader_basic_test(self.FLAGS.data, self.FLAGS.rgb_file)
        self.train_loader = torch.utils.data.DataLoader(train_dataset,
                                                        self.FLAGS.batch_size, shuffle=True,
                                                        num_workers=self.FLAGS.workers)

    def train(self):
        """
        Train the Network
        """
        for step in range(self.FLAGS.iterations):
            train_loss, train_correct = 0.0, 0.0
            test_loss, test_correct = 0.0, 0.0

            for i, data in enumerate(self.train_loader, 0):
                # get the inputs
                images, labels = data
                loss, correct = self.model.train_step(images, labels, forward_only=False)
                train_loss += loss
                train_correct += correct

            for test_index in range(len(self.test_dataset)):
                test_image, test_labels = self.test_dataset[test_index]
                loss, correct = self.model.test_step(test_image, test_labels)
