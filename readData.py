import numpy as np
import scipy.io as io
import os
import pickle
import cv2
from multiprocessing import Pool
import time
import torchvision.transforms as trans


def process_data_from_mat():
    # Read data from mat
    data_dir = '/home/yongyi/ucf101_train/my_code/data'
    name = io.loadmat(os.path.join(data_dir, 'name.mat'))['name']
    label = io.loadmat(os.path.join(data_dir, 'label.mat'))['label']
    one_hot_label = io.loadmat(os.path.join(data_dir, 'one_hot_label.mat'))['one_hot_label']
    data_set = io.loadmat(os.path.join(data_dir, 'set.mat'))['set']
    nFrames = io.loadmat(os.path.join(data_dir, 'nFrames.mat'))['nFrames']

    nObject = name.shape[1]
    train_nObject = np.sum(data_set == 1)
    test_nObject = np.sum(data_set == 2)

    name_list = []
    for i in xrange(nObject):
        a = name[0, i].tolist()
        b = str(a[0])  # example: 'v_ApplyEyeMakeup_g08_c01.avi'
        fname, back = b.split('.')
        name_list.append(fname)  # example: 'v_ApplyEyeMakeup_g08_c01'

    train_name = name_list[:train_nObject]
    test_name = name_list[train_nObject:]

    train_label = one_hot_label[:train_nObject, :]
    test_label = one_hot_label[train_nObject:, :]

    train_nFrames = nFrames[0, :train_nObject]
    test_nFrames = nFrames[0, train_nObject:]

    train_index = label[0, :train_nObject]
    test_index = label[0, train_nObject:]

    f = open(os.path.join(data_dir, 'train_name.pkl'), 'wb')
    pickle.dump(train_name, f)
    f.close()
    f = open(os.path.join(data_dir, 'test_name.pkl'), 'wb')
    pickle.dump(test_name, f)
    f.close()
    f = open(os.path.join(data_dir, 'train_label.pkl'), 'wb')
    pickle.dump(train_label, f)
    f.close()
    f = open(os.path.join(data_dir, 'test_label.pkl'), 'wb')
    pickle.dump(test_label, f)
    f.close()
    f = open(os.path.join(data_dir, 'train_nFrames.pkl'), 'wb')
    pickle.dump(train_nFrames, f)
    f.close()
    f = open(os.path.join(data_dir, 'test_nFrames.pkl'), 'wb')
    pickle.dump(test_nFrames, f)
    f.close()
    f = open(os.path.join(data_dir, 'train_index.pkl'), 'wb')
    pickle.dump(train_index, f)
    f.close()
    f = open(os.path.join(data_dir, 'test_index.pkl'), 'wb')
    pickle.dump(test_index, f)
    f.close()


def random_color_jittering(img):
    process_flag = np.random.randint(2)
    if process_flag == 0:
        jittered_img = np.zeros(img.shape, dtype=np.uint8)
        jittered_img[:, :, 0] = img[:, :, np.random.randint(3)]
        jittered_img[:, :, 1] = img[:, :, np.random.randint(3)]
        jittered_img[:, :, 2] = img[:, :, np.random.randint(3)]
    else:
        jittered_img = img
    return jittered_img


def random_flipping(img):
    process_flag = np.random.randint(2)
    if process_flag == 0:
        flipped_img = cv2.flip(img, 1)
    else:
        flipped_img = img
    return flipped_img


def img_resize_256(img):
    h, w = img.shape[0], img.shape[1]

    if h <= w:
        h_pi = int((h * 256.0) / h)
        w_pi = int((w * 256.0) / h)
    else:
        h_pi = int((h * 256.0) / w)
        w_pi = int((w * 256.0) / w)

    img_resized = cv2.resize(img, (w, h))
    return img_resized


def random_cropping(img):
    processing_flag = np.random.randint(7)
    if processing_flag == 0:
        img_crop = img[0:224, 0:224]
    elif processing_flag == 1:
        img_crop = img[0:224, img.shape[1] - 224:img.shape[1]]
    elif processing_flag == 2:
        img_crop = img[img.shape[0] - 224:img.shape[0], 0:224]
    elif processing_flag == 3:
        img_crop = img[img.shape[0] - 224:img.shape[0], img.shape[1] - 224:img.shape[1]]
    else:
        center_h = img.shape[0] / 2
        center_w = img.shape[1] / 2
        img_crop = img[center_h - 112:center_h + 112, center_w - 112:center_w + 112]
    return img_crop


def extract_image_from_file(file_name, nFrames, data_args=True):
    """
    Read image from file, including random sample a frame in a video.
    Input:
        file_name: example: 'v_ApplyEyeMakeup_g08_c01'
        nFrames
    Output:
        image: with size (224, 224, 3), np.float32
    """
    data_dir = '/home/share/ucf-data/jpegs_256'
    n = np.random.randint(1, nFrames + 1)
    image_path = os.path.join(data_dir, file_name, ('frame' + '%06d' % n + '.jpg'))
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.cv.CV_BGR2RGB)  # cv2 default color code is BGR

    if data_args:
        image = trans.Compose([trans.ToPILImage(), trans.Scale(256),
                               trans.RandomCrop(224), trans.RandomHorizontalFlip(), trans.ToTensor(),
                               trans.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])(image)
    else:
        image = trans.Compose([trans.ToPILImage(), trans.Scale(256),
                               trans.CenterCrop(224), trans.ToTensor(),
                               trans.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])(image)

    image = image.numpy()
    image = np.transpose(image, [1, 2, 0])

    # image = img_resize_256(image)
    # image = random_cropping(image)
    # # Do some processing
    # if data_args:
    #     image = random_color_jittering(image)
    #     image = random_flipping(image)

    # mean = np.asarray([0.485, 0.456, 0.406])
    # # std = np.asarray([0.229, 0.224, 0.225])
    # # image = cv2.resize(image, (224,224))
    # image = image/255.0
    # # image = (image-mean)/std
    # image = (image-mean)
    return image  # with size 224*224*3


def extract_flow_from_file(file_name, nFrames):
    """
    Read image from file, including random sample a frame in a video and construct 20-channel flow.
    Input:
        file_name: example: 'v_ApplyEyeMakeup_g08_c01'
        nFrames
    Output:
        image: with size (224, 224, 20), np.float32
    """
    data_u_dir = '/home/share/ucf-data/tvl1_flow/u'
    data_v_dir = '/home/share/ucf-data/tvl1_flow/v'
    n = np.random.randint(5, nFrames - 5)  # range 5 to nFrames-4 since we need at least 10 frames to construct flow
    flow = np.zeros((224, 224, 20), dtype=np.uint8)

    for i in range(-4, 6):
        image_u_path = os.path.join(data_u_dir, file_name, ('frame' + '%06d' % (n + i) + '.jpg'))
        image_v_path = os.path.join(data_v_dir, file_name, ('frame' + '%06d' % (n + i) + '.jpg'))
        image_u = cv2.imread(image_u_path, cv2.cv.CV_LOAD_IMAGE_GRAYSCALE)
        image_v = cv2.imread(image_v_path, cv2.cv.CV_LOAD_IMAGE_GRAYSCALE)
        image_u = img_resize_256(image_u)
        image_v = img_resize_256(image_v)
        image_u = random_cropping(image_u)
        image_v = random_cropping(image_v)

        # image_u = cv2.resize(image_u, (224,224))
        # image_v = cv2.resize(image_v, (224,224))
        image_u = (1 - image_u / 255.0) - 0.5
        image_v = image_v / 255.0 - 0.5

        flow[:, :, (i + 4) * 2] = image_u
        flow[:, :, (i + 4) * 2 + 1] = image_v

    return flow  # with size 224*224*20

# Functions to make parallel reading works


def read_next_data(data_name, nFrames, label, data_args=True):
    image = extract_image_from_file(data_name, nFrames, data_args=data_args)
    return image, label


def read_next_data_warpper(args):
    # time.sleep(10)
    return read_next_data(*args)


def read_next_flow(data_name, nFrames, label):
    image = extract_flow_from_file(data_name, nFrames)
    return image, label


def read_next_flow_warpper(args):
    # time.sleep(10)
    return read_next_flow(*args)


class DataSet(object):
    """docstring for DataSet"""

    def __init__(self, dataset, batch_size=256, shuffle=True, num_class=101, data_type='rgb', data_dir='/home/yongyi/ucf101_train/my_code/data'):
        self.data_dir = data_dir
        self.data_name, self.label, self.nFrames, self.index = self.read_file(dataset)
        self.num_example = len(self.data_name)
        self.num_class = num_class
        self.batch_size = batch_size
        self.batch_int = batch_size / self.num_class
        self.batch_res = batch_size % self.num_class
        self.shuffle = shuffle
        self.index_list = []  # the index of example of each class
        self.max_index_num = []  # the number of example of each class
        for i in range(1, 102):
            tmp = np.where(self.index == i)[0]
            self.index_list.append(tmp)
            self.max_index_num.append(tmp.shape[0])

        self.current_index = [0] * self.num_example

        self.next_batch_index = 0  # to control self.next_batch()
        if data_type == 'flow':
            self.read_next_func = read_next_flow_warpper
        else:
            self.read_next_func = read_next_data_warpper
        self.epoch = 0
        self.epoch_count = 0

    def read_file(self, dataset):
        if dataset == 'train':
            f = open(os.path.join(self.data_dir, 'train_name.pkl'), 'r')
            data_name = pickle.load(f)
            f.close()
            f = open(os.path.join(self.data_dir, 'train_label.pkl'), 'r')
            label = pickle.load(f)
            f.close()
            f = open(os.path.join(self.data_dir, 'train_nFrames.pkl'), 'r')
            nFrames = pickle.load(f)
            f.close()
            f = open(os.path.join(self.data_dir, 'train_index.pkl'), 'r')
            index = pickle.load(f)
            f.close()
        else:
            f = open(os.path.join(self.data_dir, 'test_name.pkl'), 'r')
            data_name = pickle.load(f)
            f.close()
            f = open(os.path.join(self.data_dir, 'test_label.pkl'), 'r')
            label = pickle.load(f)
            f.close()
            f = open(os.path.join(self.data_dir, 'test_nFrames.pkl'), 'r')
            nFrames = pickle.load(f)
            f.close()
            f = open(os.path.join(self.data_dir, 'test_index.pkl'), 'r')
            index = pickle.load(f)
            f.close()
        return data_name, label, nFrames, index

    def next_batch_train(self):
        batch_image = np.zeros((self.batch_size, 224, 224, 3))
        batch_label = np.zeros((self.batch_size, self.num_class))
        extract_len = [self.batch_int] * self.num_class
        random_sample = np.random.randint(0, self.num_class, [self.batch_res])
        for i in range(self.batch_res):
            extract_len[random_sample[i]] += 1

        this_batch_index = 0
        """Using parallel to speed up"""
        for i in range(len(extract_len)):
            for j in range(extract_len[i]):
                index = self.current_index[i]
                data_index = self.index_list[i][index]

                image = extract_image_from_file(self.data_name[data_index], self.nFrames[data_index])
                label = self.label[data_index]
                # print image.shape
                batch_image[this_batch_index] = image
                batch_label[this_batch_index] = label
                this_batch_index += 1
                self.current_index[i] += 1
                self.epoch_count += 1
                if self.epoch_count % self.num_example == 0:
                    self.epoch += 1
                    self.epoch_count = 0
                if self.current_index[i] == self.max_index_num[i]:
                    # shuffle
                    perm = np.arange(self.index_list[i].shape[0])
                    np.random.shuffle(perm)
                    self.index_list[i] = self.index_list[i][perm]
                    self.current_index[i] = 0
        perm = np.arange(batch_image.shape[0])
        np.random.shuffle(perm)
        batch_image = batch_image[perm]
        batch_label = batch_label[perm]

        return batch_image, batch_label

    def next_batch_train_parallel(self):
        """
        Read training data in a parallel way which is ten times faster
        """
        extract_len = [self.batch_int] * self.num_class
        random_sample = np.random.randint(0, self.num_class, [self.batch_res])
        for i in range(self.batch_res):
            extract_len[random_sample[i]] += 1
        # Get the list of data params
        this_batch_index = 0
        batch_name_list = []
        batch_nFrames_list = []
        batch_label_list = []
        """Using parallel to speed up"""
        for i in range(len(extract_len)):
            for j in range(extract_len[i]):
                index = self.current_index[i]
                data_index = self.index_list[i][index]
                batch_name_list.append(self.data_name[data_index])
                batch_nFrames_list.append(self.nFrames[data_index])
                batch_label_list.append(self.label[data_index])
                this_batch_index += 1
                self.current_index[i] += 1
                self.epoch_count += 1
                if self.epoch_count % self.num_example == 0:
                    self.epoch += 1
                    self.epoch_count = 0
                if self.current_index[i] == self.max_index_num[i]:
                    # self.epoch += 1
                    # shuffle
                    perm = np.arange(self.index_list[i].shape[0])
                    np.random.shuffle(perm)
                    self.index_list[i] = self.index_list[i][perm]
                    self.current_index[i] = 0
        # Read data using multiprocessing
        data_args = [True] * len(batch_name_list)
        p = Pool(processes=4)
        try:
            results = p.map_async(self.read_next_func, zip(batch_name_list, batch_nFrames_list, batch_label_list, data_args)).get(9999999)
            p.close()
            p.join()
        except KeyboardInterrupt:
            print 'ctrl + c'
            p.terminate()
            p.join()
        results = zip(*results)
        image_list = results[0]
        label_list = results[1]
        # From list to array
        batch_image = np.stack(image_list)
        batch_label = np.vstack(label_list)
        # shuffle data
        perm = np.arange(batch_image.shape[0])
        np.random.shuffle(perm)
        batch_image = batch_image[perm]
        batch_label = batch_label[perm]
        return batch_image, batch_label

    def next_batch(self, in_batch_size):
        batch_image = np.zeros((in_batch_size, 224, 224, 3))
        batch_label = np.zeros((in_batch_size, self.num_class))

        for i in range(in_batch_size):
            image = extract_image_from_file(self.data_name[self.next_batch_index], self.nFrames[self.next_batch_index])
            label = self.label[self.next_batch_index]
            batch_image[i] = image
            batch_label[i] = label
            self.next_batch_index += 1
            if self.next_batch_index > self.num_example:
                self.next_batch_index = 0
        return batch_image, batch_label

    def next_batch_parallel(self, in_batch_size):
        """
        Read testing data in a parallel way which is ten times faster
        """
        # Get the list of data params
        batch_name_list = []
        batch_nFrames_list = []
        batch_label_list = []
        for i in range(in_batch_size):
            batch_name_list.append(self.data_name[self.next_batch_index])
            batch_nFrames_list.append(self.nFrames[self.next_batch_index])
            batch_label_list.append(self.label[self.next_batch_index])
            self.next_batch_index += 1
            if self.next_batch_index >= self.num_example:
                self.next_batch_index = 0
        # Read data using multiprocessing
        data_args = [False] * len(batch_name_list)
        p = Pool(processes=4)
        try:
            results = p.map_async(self.read_next_func, zip(batch_name_list, batch_nFrames_list, batch_label_list, data_args)).get(9999999)
            p.close()
            p.join()
        except KeyboardInterrupt:
            print 'ctrl + c'
            p.terminate()
            p.join()
        results = zip(*results)
        image_list = results[0]
        label_list = results[1]
        batch_image = np.stack(image_list)
        batch_label = np.vstack(label_list)
        return batch_image, batch_label
