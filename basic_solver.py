from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np
from utilities import save_checkpoint, read_checkpoint
from logger import Logger
import time
from video_data_loader import ucf101_rgb_loader_basic_train, ucf101_rgb_loader_basic_test
from video_data_loader import ucf101_flow_loader_basic_train, ucf101_flow_loader_basic_test
from model import resnet18_basic


class solver(object):
    def __init__(self, FLAGS, train_dir, summaries_dir):
        self.FLAGS = FLAGS
        self.summaries_dir = summaries_dir
        self.logger = Logger(self.summaries_dir)
        self.train_dir = train_dir
        self.data_type = FLAGS.data_type
        self.model = self.create_model()
        self.creat_data_loader()

    def create_model(self, sampling=False):
        """Create translation model and initialize or load parameters in session."""

        model = resnet18_basic(self.FLAGS, self.data_type)

        if not self.FLAGS.resume:
            print("Creating model with fresh parameters.")
            self.start_step = 0
        else:
            print(self.train_dir)
            model_exist, checkpoint = read_checkpoint(self.train_dir)
            if model_exist:
                model.model.load_state_dict(checkpoint['state_dict'])
                self.start_step = checkpoint['step'] + 1
                model.learning_rate = checkpoint['lr']
                model.set_optimizer(model.learning_rate, 1.0)

        return model

    def creat_data_loader(self):
        if self.data_type == 'rgb':
            train_dataset = ucf101_rgb_loader_basic_train(self.FLAGS.data, self.FLAGS.rgb_file)
            test_dataset = ucf101_rgb_loader_basic_test(self.FLAGS.data, self.FLAGS.rgb_file)
        elif self.data_type == 'flow':
            train_dataset = ucf101_flow_loader_basic_train(self.FLAGS.data, self.FLAGS.flow_file)
            test_dataset = ucf101_flow_loader_basic_test(self.FLAGS.data, self.FLAGS.flow_file)
        else:
            raise('Error data type: ', self.data_type)
        self.train_loader = torch.utils.data.DataLoader(train_dataset,
                                                        self.FLAGS.batch_size, shuffle=True,
                                                        num_workers=self.FLAGS.workers)
        self.test_loader = torch.utils.data.DataLoader(test_dataset,
                                                       1, shuffle=False,
                                                       num_workers=self.FLAGS.workers)

    def train(self):
        """
        Train the Network
        """
        for step in range(self.start_step, self.FLAGS.iterations):
            train_loss, train_correct = [], []
            test_loss, test_correct = [], []

            start_time = time.time()
            for i, data in enumerate(self.train_loader, 0):
                # get the inputs
                images, labels = data
                loss, correct = self.model.train_step(images, labels, forward_only=False)
                train_loss.append(loss.numpy())
                train_correct.append(correct.numpy())
            total_train_loss = np.mean(np.hstack(train_loss))
            total_train_correct = np.mean(np.hstack(train_correct))

            step_time = (time.time() - start_time)

            print("============================\n"
                  "Data Type:           %s\n"
                  "Global step:         %d\n"
                  "Learning rate:       %.4f\n"
                  "Step-time (ms):      %.4f\n"
                  "Train loss avg:      %.4f\n"
                  "Train Accuracy:      %.4f\n"
                  "============================" % (self.data_type, step + 1,
                                                    self.model.learning_rate, step_time * 1000,
                                                    total_train_loss, total_train_correct))
            if (step + 1) % 100 == 0:
                start_time = time.time()
                for i, data in enumerate(self.test_loader, 0):
                    images, labels = data
                    test_image, test_labels = images[0, :], labels[0, :]
                    # test_image, test_labels = self.test_dataset[test_index]
                    loss, correct = self.model.test_step(test_image, test_labels)
                    test_loss.append(loss.numpy())
                    test_correct.append(correct.numpy())
                total_test_loss = np.mean(np.hstack(test_loss))
                total_test_correct = np.mean(np.hstack(test_correct))
                step_time = (time.time() - start_time)
                print("Test-time (ms):     %.4f\n"
                      "Test loss avg:      %.4f\n"
                      "Test Accuracy:      %.4f\n"
                      "============================" % (step_time * 1000,
                                                        total_test_loss, total_test_correct))
                self.logger.scalar_summary('test_loss', total_test_loss, step + 1)
                self.logger.scalar_summary('test_acc', total_test_correct, step + 1)
            print()

            self.logger.scalar_summary('train_loss', total_train_loss, step + 1)
            self.logger.scalar_summary('train_acc', total_train_correct, step + 1)

            self.logger.scalar_summary('learning_rate', self.model.learning_rate, step + 1)

            # Adjust Learning Rate
            if (step + 1) == 100:  # Unfreeze the parameters
                self.model.set_optimizer(self.model.learning_rate, 1.0)
            if (step + 1) % self.FLAGS.learning_rate_step == 0:
                self.model.learning_rate = self.model.learning_rate * self.FLAGS.learning_rate_decay_factor
                self.model.set_optimizer(self.model.learning_rate, 1.0)
                # Save Checkpoint
            if (step + 1) % self.FLAGS.print_freq == 0:
                print("Saving the model...")
                start_time = time.time()
                save_checkpoint({
                    'step': step,
                    'lr': self.model.learning_rate,
                    'state_dict': self.model.model.state_dict()
                }, self.train_dir, 100)
                print('Saving checkpoint at step: %d' % (step + 1))
                # model.saver.save(sess, os.path.normpath(os.path.join(train_dir, 'checkpoint')), global_step=current_step )
                print("done in {0:.2f} ms".format((time.time() - start_time) * 1000))
