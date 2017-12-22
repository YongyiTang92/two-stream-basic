import argparse
import os
import torch
import torch.backends.cudnn as cudnn
from basic_solver import solver


def main(FLAGS, train_dir, summaries_dir):
  cudnn.benchmark = True
  os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.device

  solver_model = solver(FLAGS, train_dir, summaries_dir)
  solver_model.train()
  # if FLAGS.sample:
  #     solver_model.sample()
  # else:
  #     solver_model.train()


parser = argparse.ArgumentParser(description='PyTorch UCF101 Training')
# Paths
parser.add_argument('--data', default='/home/yongyi/ucf101_train/my_code/data', type=str, help='path to pkls')
parser.add_argument('--rgb_file', default='/data/yongyi/ucf101_data/jpegs_256', type=str, help='path to dataset')
parser.add_argument('--flow_file', default='/data/yongyi/ucf101_data/tvl1_flow', type=str, help='path to dataset')
parser.add_argument('--train_dir', default='./experiments', type=str,
                    help='path to latest checkpoint (default: none)')

# Parameters
parser.add_argument('--workers', default=4, type=int,
                    help='number of data loading workers (default: 4)')
parser.add_argument('--iterations', default=5000, type=int,
                    help='number of total iterations to run')
parser.add_argument('--start_epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--print_freq', default=500, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--device', default='0', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--resume', help="Load the check points.", action="store_true")
parser.add_argument('--data_type', default='rgb', type=str, help='input data type: rgb or flow(5-frames)')
parser.add_argument('--model', default='resnet18', type=str, help='model type')

# Hyper-Parameters
parser.add_argument('--optimizer', default='SGD', type=str, help='Type of optimizer')
parser.add_argument('--batch_size', default=256, type=int,
                    help='mini-batch size (default: 256)')
parser.add_argument('--lr', default=0.001, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='momentum')
parser.add_argument('--weight_decay', default=1e-4, type=float,
                    help='weight decay (default: 1e-4)')
parser.add_argument('--learning_rate_decay_factor', default=0.1, type=float,
                    help='learning_rate_decay_factor (default: 0.1)')
parser.add_argument('--learning_rate_step', default=1500, type=int,
                    help='learning_rate_step (default: 1500)')
parser.add_argument('--max_gradient_norm', default=40, type=float,
                    help='max_gradient_norm (default: 40)')
parser.add_argument('--test_segs', default=5, type=int,
                    help='number of segments for testing (default: 5)')
parser.add_argument('--pooling', default='mean', type=str,
                    help='pooling method for testing')


# parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")

FLAGS = parser.parse_args()
print(FLAGS)
train_dir = os.path.normpath(os.path.join(FLAGS.train_dir,
                                          'iterations_{0}'.format(FLAGS.iterations),
                                          'batch_size_{0}'.format(FLAGS.batch_size),
                                          'lr_{0}'.format(FLAGS.lr),
                                          'momentum_{0}'.format(FLAGS.momentum),
                                          'learning_rate_step_{0}'.format(FLAGS.learning_rate_step),
                                          'learning_rate_decay_factor_{0}'.format(FLAGS.learning_rate_decay_factor),
                                          'test_segs_{0}'.format(FLAGS.test_segs),
                                          FLAGS.pooling,
                                          FLAGS.optimizer,
                                          FLAGS.model,
                                          FLAGS.data_type,
                                          'weight_decay{0}'.format(FLAGS.weight_decay)))

summaries_dir = os.path.normpath(os.path.join(train_dir, "log"))  # Directory for TB summaries
main(FLAGS, train_dir, summaries_dir)
