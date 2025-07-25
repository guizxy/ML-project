import numpy as np


class FullyConnectedLayer(object):  # 全连接层
    def __init__(self, num_input, num_output):  # 初始化
        self.num_input = num_input
        self.num_output = num_output
        print('\tFully connected layer initialized with input %d, output %d.' %
              (self.num_input, self.num_output))

    def init_param(self, std=0.01):    # 参数初始化
        self.weight = np.random.normal(
            loc=0.0, scale=std, size=(self.num_input, self.num_output))
        self.bias = np.zeros([1, self.num_output])

    def forward(self, input):  # 全连接层前向传播
        self.input = input
        self.output = np.matmul(input, self.weight) + self.bias
        return self.output

    def backward(self, top_diff):  # 全连接层反向传播
        self.d_weight = np.matmul(self.input.T, top_diff)
        self.d_bias = np.matmul(np.ones([1, top_diff.shape[0]]), top_diff)
        bottom_diff = np.matmul(top_diff, self.weight.T)
        return bottom_diff

    def load_param(self, weight, bias):  # 加载新参数
        assert self.weight.shape == weight.shape
        assert self.bias.shape == bias.shape
        self.weight = weight
        self.bias = bias


class ReLULayer(object):
    def __init__(self):
        print('\tReLU layer.')

    def forward(self, input):  # ReLU激活函数前向传播
        self.input = input
        output = np.maximum(self.input, 0)
        return output

    def backward(self, top_diff):  # ReLU激活函数反向传播
        bottom_diff = top_diff * (self.input > 0)
        return bottom_diff


class SoftmaxLossLayer(object):
    def __init__(self):
        print('\tSoftmax loss layer.')

    def forward(self, input):  # Softmax前向传播
        input_max = np.max(input, axis=1, keepdims=True)
        input_exp = np.exp(input-input_max)
        self.prob = input_exp / np.sum(input_exp, axis=1, keepdims=True)
        return self.prob

    def get_loss(self, label):  # 计算损失
        self.batch_size = self.prob.shape[0]
        self.label_onehot = np.zeros_like(self.prob)
        self.label_onehot[np.arange(self.batch_size), label] = 1.0
        loss = -np.sum(np.log(self.prob) * self.label_onehot) / self.batch_size
        return loss

    def backward(self):  # Softmax反向传播
        bottom_diff = (self.prob - self.label_onehot) / self.batch_size
        return bottom_diff
