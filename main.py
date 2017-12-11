import argparse
import os
import shutil
import time
import numpy as np
from readData import DataSet
import multiprocessing
import sys
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch.autograd import Variable
from my_resnet import resnet18, resnet50, resnet50_flow

parser = argparse.ArgumentParser(description='PyTorch UCF101 Training')
parser.add_argument('--data', default='/home/yongyi/ucf101_train/my_code/data', type=str, metavar='DIR', help='path to dataset')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100000, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch_size', '--batch_size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning_rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print_freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='/home/yongyi/ucf101_train/my_code/data/', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--device', default='0', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')


def main():
    args = parser.parse_args()
    print('creat model')
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    model_tmp = resnet50(pretrained=True)
    best_prec1 = 0.0
    if args.resume:
        is_exist, checkpoint = read_checkpoint(args.resume)
        if is_exist:
            # print("=> loading checkpoint '{}'".format(args.resume))
            # checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model_tmp.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint (epoch {})"
                  .format(checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # cudnn.benchmark = True # ???

    # Load data
    print('Loading dataset')
    train_set = DataSet('train', batch_size=args.batch_size)
    test_set = DataSet('test')
    test_example = test_set.num_example
    train_set.epoch = args.start_epoch

    if torch.cuda.is_available():
        model_tmp.cuda()
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = nn.CrossEntropyLoss()

    lr = args.lr
    # optimizer = torch.optim.SGD(model_tmp.parameters(), lr, momentum=args.momentum,
    #                             weight_decay=args.weight_decay)

    ignored_params = list(map(id, model_tmp.fc_new.parameters()))
    base_params = filter(lambda p: id(p) not in ignored_params, model_tmp.parameters())
    optimizer = torch.optim.SGD(
        [{'params': base_params}, {'params': model_tmp.fc_new.parameters(), 'lr': lr}],
        lr=lr * 0.1, momentum=args.momentum, weight_decay=args.weight_decay)

    #device_ids = [int(x) for x in args.device.split(',')]
    # model = torch.nn.DataParallel(model_tmp, device_ids=device_ids) # use parallel training
    model = torch.nn.DataParallel(model_tmp, [0])  # use parallel training

    if args.resume:
        if is_exist:
            print ('Last checkpoint testing:')
            test_acc = test_acc_func(model, test_set, train_set.epoch, args.batch_size)

    print('start training')
    step = 0
    while train_set.epoch < args.epochs:
        start_time = time.time()
        train_img, train_label = train_set.next_batch_train_parallel()
        train_img = train_img.transpose((0, 3, 1, 2))
        train_img, train_label = torch.from_numpy(train_img), torch.from_numpy(train_label)
        train_img, train_label = train_img.float(), train_label.float()
        train_img_var, train_label_var = Variable(train_img), Variable(train_label)
        if torch.cuda.is_available():
            train_img_var, train_label_var = train_img_var.cuda(), train_label_var.cuda()

        _, gt = torch.max(train_label_var, 1)
        middle_time = time.time()
        model.train()
        train_out = model.forward(train_img_var)
        loss = criterion(train_out, gt[:, 0])

        _, pred = torch.max(train_out, 1)

        correct = pred.eq(gt)
        accuracy = correct.float().sum(0).mul(100.0 / train_img_var.size(0))
        losses = loss.float()

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        end_time = time.time()

        if step % args.print_freq == 0:
            print('Epoch: {},'
                  'Step: {},'
                  'Loading Time: {:.3f} s'
                  'Time: {:.3f} s'
                  'Base Lr: {:.4f}'
                  'Training Acc: {:.3f}%,'
                  'Loss: {:.4f}'.format(
                      train_set.epoch, step, middle_time - start_time,
                      end_time - middle_time, lr,
                      accuracy.data[0, 0], losses.data[0]
                  )
                  )

        if step % 20 * args.print_freq == 0 and step > 10:
            test_acc = test_acc_func(model, test_set, train_set.epoch, args.batch_size)
            if test_acc > best_prec1:
                is_best = 1
                best_prec1 = test_acc
                save_checkpoint({
                    'epoch': train_set.epoch + 1,
                    'state_dict': model_tmp.state_dict(),
                    'best_prec1': best_prec1,
                }, is_best, args.data)

        if step == 10000:
            lr = lr / 2
            optimizer = torch.optim.SGD(
                [{'params': base_params}, {'params': model_tmp.fc_new.parameters(), 'lr': lr}],
                lr=lr * 0.1, momentum=args.momentum, weight_decay=args.weight_decay)
        if step == 30000:
            lr = lr / 2
            optimizer = torch.optim.SGD(
                [{'params': base_params}, {'params': model_tmp.fc_new.parameters(), 'lr': lr}],
                lr=lr * 0.1, momentum=args.momentum, weight_decay=args.weight_decay)

        step += 1


def test_acc_func(model, test_set, epoch, batch_size):
    test_acc = 0.0
    for i in range(int(np.ceil(float(test_set.num_example) / batch_size))):
        test_batch = min((i + 1) * batch_size, test_set.num_example) - i * batch_size
        # print test_batch
        test_img, test_label = test_set.next_batch_parallel(test_batch)
        test_img = test_img.transpose((0, 3, 1, 2))
        test_img, test_label = torch.from_numpy(test_img), torch.from_numpy(test_label)
        test_img, test_label = test_img.float(), test_label.float()
        test_img_var, test_label_var = Variable(test_img, volatile=True), Variable(test_label)
        if torch.cuda.is_available():
            test_img_var, test_label_var = test_img_var.cuda(), test_label_var.cuda()
        model.eval()
        test_out = model.forward(test_img_var)
        _, gt = torch.max(test_label_var, 1)

        _, test_pred = torch.max(test_out, 1)
        correct = test_pred.eq(gt)
        test_acc += correct.float().sum(0).data[0, 0]

    test_acc = test_acc / float(test_set.num_example)
    print('Testing Result: '
          'Epoch: {},'
          'Testing Acc: {:.3f}%,'.format(
              epoch,
              test_acc * 100
          )
          )
    return test_acc


def save_checkpoint(state, is_best, _dir, max_model=5):
    filename = 'checkpoint' + str(state['epoch']) + '.pth.tar'
    txtname = 'checkpoint.txt'
    # f = os.path.join(_dir, filename)
    # if not os.path.exists(f):
    #     os.mkdir()
    file_dir = os.path.join(_dir, filename)
    torch.save(state, file_dir)
    txt_dir = os.path.join(_dir, txtname)

    with open(txt_dir, 'a') as f_txt:
        f_txt.write(filename + '\n')

    with open(txt_dir, 'r') as f_txt:
        f_lines = f_txt.readlines()
        tmp_lines = [a.strip('\n') for a in f_lines]
        if len(f_lines) > max_model:
            tmp_name = tmp_lines.pop(0)
            f_lines.pop(0)
            if os.path.isfile(os.path.join(_dir, tmp_name)):
                os.remove(os.path.join(_dir, tmp_name))

    with open(txt_dir, 'w') as f_txt:
        write_names = ''.join(f_lines)
        f_txt.write(write_names)

    # if is_best:
    #     shutil.copyfile(filename, 'model_best.pth.tar')


def read_checkpoint(_dir):
    txtname = 'checkpoint.txt'
    txt_dir = os.path.join(_dir, txtname)
    if not os.path.isfile(txt_dir):
        checkpoint = None
        is_exist = False
        return is_exist, checkpoint

    with open(txt_dir, 'a+') as f_txt:
        f_lines = f_txt.readlines()
        tmp_lines = [a.strip('\n') for a in f_lines]
        name = tmp_lines[-1]

    checkpoint_file = os.path.join(_dir, name)
    if os.path.isfile(checkpoint_file):
        print('Reading checkpoint file: ' + name)
        checkpoint = torch.load(checkpoint_file)
        is_exist = True
    else:
        print ('Checkpoint file ' + name + ' not exist!')
        checkpoint = None
        is_exist = False
    return is_exist, checkpoint


def myexcepthook(exctype, value, traceback):
    for p in multiprocessing.active_children():
        p.terminate()


if __name__ == '__main__':
    main()
    sys.excepthook = myexcepthook
    sleep(1)
