import numpy as np
import tqdm
import random
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import os
import struct

def sigmoid(x): # 激活函数采用Sigmoid
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):	# Sigmoid的导数
    return sigmoid(x) * (1 - sigmoid(x))


class NeuralNetwork:	# 神经网络
    def __init__(self, layers):	# layers为神经元个数列表
        self.activation = sigmoid	# 激活函数
        self.activation_deriv = sigmoid_derivative	# 激活函数导数
        self.weights = []	# 权重列表
        self.bias = []	# 偏置列表
        for i in range(1, len(layers)):	# 正态分布初始化
            self.weights.append(np.random.randn(layers[i-1], layers[i]))
            self.bias.append(np.random.randn(layers[i]))

    def fit(self, x, y, learning_rate=0.2, epochs=5):	# 反向传播算法
        x = np.atleast_2d(x)
        n = len(y)	# 样本数
        y = np.array(y)
        loss=[]
        for k in range(epochs * 10000):	# 带进度条的训练过程
            print(k)
            a = [x[k]]	# 保存各层激活值的列表
            # 正向传播开始
            for lay in range(len(self.weights)):
                a.append(self.activation(np.dot(a[lay], self.weights[lay]) + self.bias[lay]))
            # 反向传播开始
            label=y[k]
            error = label - a[-1]	# 误差值
            loss.append(np.sum(error**2))
            deltas = [error * self.activation_deriv(a[-1])]	# 保存各层误差值的列表

            layer_num = len(a) - 2	# 导数第二层开始
            for j in range(layer_num, 0, -1):
                deltas.append(deltas[-1].dot(self.weights[j].T) * self.activation_deriv(a[j]))	# 误差的反向传播
            deltas.reverse()
            for i in range(len(self.weights)):	# 正向更新权值
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                self.weights[i] += learning_rate * layer.T.dot(delta)
                self.bias[i] += learning_rate * deltas[i]
        plt.plot([i for i in range(len(loss))],loss)
        plt.savefig('./train2.png')
        plt.show()
    def predict(self, x):	# 预测
        a = np.array(x, dtype=np.float)
        for lay in range(0, len(self.weights)):	# 正向传播
            a = self.activation(np.dot(a, self.weights[lay]) + self.bias[lay])
        a = list(100 * a/sum(a))	# 改为百分比显示
        i = a.index(max(a))	# 预测值
        per = []	# 各类的置信程度
        for num in a:
            per.append(str(round(num, 2))+'%')
        return i, per

def onehot(targets):
    num = len(targets)
    result = np.zeros((num, 10))
    for i in range(num):
        result[i][targets[i]] = 1
    return result

def load_mnist(path='./mnist', kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path, '%s-labels.idx1-ubyte' % kind)

    images_path = os.path.join(path, '%s-images.idx3-ubyte' % kind)

    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)
    # 读入magic是一个文件协议的描述,也是调用fromfile 方法将字节读入NumPy的array之前在文件缓冲中的item数(n).

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
    labels=onehot(labels)
    return images, labels

if __name__ == '__main__':

    mynet = NeuralNetwork([784,128,10])
    train_image, train_labels = load_mnist(path='./mnist', kind='train')
    test_image, test_labels = load_mnist(path='./mnist', kind='test')

    mynet.fit(train_image,train_labels)

