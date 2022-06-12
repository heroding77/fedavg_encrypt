import numpy as np
import gzip
import os
from sklearn.utils import shuffle
from torchvision import datasets, transforms
import torch



class GetDataSet(object):
    def __init__(self, dataSetName, isIID):
        self.name = dataSetName
        self.train_data = None
        self.train_label = None
        self.train_data_size = None
        self.test_data = None
        self.test_label = None
        self.test_data_size = None

        self._index_in_train_epoch = 0

        if self.name == 'mnist':
            self.mnistDataSetConstruct(isIID)
        else:
            self.cifarDataSetConstruct(isIID)


    def mnistDataSetConstruct(self, isIID):
        data_dir = r'./data/MNIST'
        # 选定图片路径
        train_images_path = os.path.join(data_dir, 'train-images-idx3-ubyte.gz')
        train_labels_path = os.path.join(data_dir, 'train-labels-idx1-ubyte.gz')
        test_images_path = os.path.join(data_dir, 't10k-images-idx3-ubyte.gz')
        test_labels_path = os.path.join(data_dir, 't10k-labels-idx1-ubyte.gz')
        # 从.gz中提取图片
        train_images = extract_images(train_images_path)
        train_labels = extract_labels(train_labels_path)
        test_images = extract_images(test_images_path)
        test_labels = extract_labels(test_labels_path)

        assert train_images.shape[0] == train_labels.shape[0]
        assert test_images.shape[0] == test_labels.shape[0]

        self.train_data_size = train_images.shape[0]
        self.test_data_size = test_images.shape[0]

        # mnist黑白图片通道为1
        assert train_images.shape[3] == 1
        assert test_images.shape[3] == 1
        # 图片展平
        train_images = train_images.reshape(train_images.shape[0], train_images.shape[1] * train_images.shape[2])
        test_images = test_images.reshape(test_images.shape[0], test_images.shape[1] * test_images.shape[2])
        
        # 标准化处理
        train_images = train_images.astype(np.float32)
        train_images = np.multiply(train_images, 1.0 / 255.0)
        test_images = test_images.astype(np.float32)
        test_images = np.multiply(test_images, 1.0 / 255.0)

        # 是否独立同分布
        if isIID:
            # 打乱顺序
            order = np.arange(self.train_data_size)
            np.random.shuffle(order)
            self.train_data = train_images[order]
            self.train_label = train_labels[order]
        else:
            # 按照0——9顺序排列
            labels = np.argmax(train_labels, axis=1)
            order = np.argsort(labels)
            self.train_data = train_images[order]
            self.train_label = train_labels[order]

        self.test_data = test_images
        self.test_label = test_labels

    def cifarDataSetConstruct(self, isIID):
        data_dir = r'./data/'
        transform_train = transforms.Compose([
			transforms.RandomCrop(32, padding=4),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
		])

        transform_test = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
		])
		
        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
										transform=transform_train)
        eval_dataset = datasets.CIFAR10(data_dir, train=False, transform=transform_test)
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=50000, shuffle=False)
        eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=10000, shuffle=False)

        train_images, train_labels = next(iter(train_loader))
        test_images, test_labels = next(iter(eval_loader))

        # 数据转换为numpy格式
        train_images = train_images.numpy()
        train_labels = dense_to_one_hot(train_labels.numpy())
        test_images = test_images.numpy()
        test_labels = dense_to_one_hot(test_labels.numpy())

        # 验证数据导入无误
        assert train_images.shape[0] == train_labels.shape[0]
        assert test_images.shape[0] == test_labels.shape[0]

        self.train_data_size = train_images.shape[0]
        self.test_data_size = test_images.shape[0]

        # cifar彩色图片通道为3
        assert train_images.shape[1] == 3
        assert test_images.shape[1] == 3
        # 图片展平
        train_images = train_images.reshape(train_images.shape[0], 3 * train_images.shape[2] * train_images.shape[3])
        test_images = test_images.reshape(test_images.shape[0], 3 * test_images.shape[2] * test_images.shape[3])

        # 是否独立同分布
        if isIID:
            # 打乱顺序
            order = np.arange(self.train_data_size)
            np.random.shuffle(order)
            self.train_data = train_images[order]
            self.train_label = train_labels[order]
        else:
            # 按照0——9顺序排列
            labels = np.argmax(train_labels, axis=1)
            order = np.argsort(labels)
            self.train_data = train_images[order]
            self.train_label = train_labels[order]

        self.test_data = test_images
        self.test_label = test_labels

# 比特流读取
def _read32(bytestream):
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]

# 提取图片
def extract_images(filename):
    """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        if magic != 2051:
            raise ValueError(
                    'Invalid magic number %d in MNIST image file: %s' %
                    (magic, filename))
        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        buf = bytestream.read(rows * cols * num_images)
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data.reshape(num_images, rows, cols, 1)
        return data

# 标签one-hot编码
def dense_to_one_hot(labels_dense, num_classes=10):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

# 提取标签
def extract_labels(filename):
    """Extract the labels into a 1D uint8 numpy array [index]."""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        if magic != 2049:
            raise ValueError(
                    'Invalid magic number %d in MNIST label file: %s' %
                    (magic, filename))
        num_items = _read32(bytestream)
        buf = bytestream.read(num_items)
        labels = np.frombuffer(buf, dtype=np.uint8)
        return dense_to_one_hot(labels)


''' 
if __name__=="__main__":
    'test data set'
    mnistDataSet = GetDataSet('mnist', True) # test NON-IID
    if type(mnistDataSet.train_data) is np.ndarray and type(mnistDataSet.test_data) is np.ndarray and \
            type(mnistDataSet.train_label) is np.ndarray and type(mnistDataSet.test_label) is np.ndarray:
        print('the type of data is numpy ndarray')
    else:
        print('the type of data is not numpy ndarray')
    print('the shape of the train data set is {}'.format(mnistDataSet.train_data.shape))
    print('the shape of the test data set is {}'.format(mnistDataSet.test_data.shape))
    print(mnistDataSet.train_label[0:100], mnistDataSet.train_label[11000:11100])
'''


'''
if __name__=="__main__":
    data = GetDataSet('cifar', 1)
'''
