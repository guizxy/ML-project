import numpy as np


class ConvolutionalLayer(object):  # 卷积层
    def __init__(self, kernel_size, channel_in, channel_out, padding, stride, type=0):
        self.kernel_size = kernel_size
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.padding = padding
        self.stride = stride
        self.forward = self.forward_raw
        self.backward = self.backward_raw
        if type == 1:  # 通过type选择加速版前向传播和反向传播函数
            self.forward = self.forward_speedup
            self.backward = self.backward_speedup
        print('\tConvolutional layer with kernel size %d, input channel %d, output channel %d.' % (
            self.kernel_size, self.channel_in, self.channel_out))

    def init_param(self, std=0.01):  # 通过高斯分布初始化参数矩阵
        self.weight = np.random.normal(loc=0.0, scale=std, size=(
            self.channel_in, self.kernel_size, self.kernel_size, self.channel_out))
        self.bias = np.zeros([self.channel_out])

    def forward_raw(self, input):  # 优化前的前向传播
        self.input = input  # [N, C, H, W]
        height = self.input.shape[2] + self.padding * 2
        width = self.input.shape[3] + self.padding * 2
        self.input_pad = np.zeros(
            [self.input.shape[0], self.input.shape[1], height, width])
        self.input_pad[:, :, self.padding:self.input.shape[2] + self.padding,
                       self.padding:self.input.shape[3] + self.padding] = self.input
        height_out = (height - self.kernel_size) // self.stride + 1
        width_out = (width - self.kernel_size) // self.stride + 1
        self.output = np.zeros(
            [self.input.shape[0], self.channel_out, height_out, width_out])
        for idxn in range(self.input.shape[0]):
            for idxc in range(self.channel_out):
                for idxh in range(height_out):
                    for idxw in range(width_out):
                        self.output[idxn, idxc, idxh, idxw] = np.sum(self.input_pad[idxn, :, idxh * self.stride: idxh * self.stride + self.kernel_size,
                                                                     idxw * self.stride: idxw * self.stride + self.kernel_size] * self.weight[:, :, :, idxc]) + self.bias[idxc]
        return self.output

    def backward_raw(self, top_diff):  # 优化前的反向传播
        self.d_weight = np.zeros(self.weight.shape)
        self.d_bias = np.zeros(self.bias.shape)
        bottom_diff = np.zeros(self.input_pad.shape)
        for idxn in range(top_diff.shape[0]):
            for idxc in range(top_diff.shape[1]):
                for idxh in range(top_diff.shape[2]):
                    for idxw in range(top_diff.shape[3]):
                        # W = W - η*(∂L/∂W), ∂L/∂W = xT * (∂L/∂z), ∂L/∂z = top_diff
                        self.d_weight[:, :, :, idxc] += top_diff[idxn, idxc, idxh, idxw] * self.input_pad[idxn, :, idxh *
                                                                                                          self.stride:idxh*self.stride+self.kernel_size, idxw*self.stride:idxw*self.stride+self.kernel_size]  # [N,H,W,C]
                        self.d_bias[idxc] += top_diff[idxn, idxc, idxh, idxw]
                        bottom_diff[idxn, :, idxh*self.stride:idxh*self.stride+self.kernel_size, idxw*self.stride:idxw *
                                    self.stride+self.kernel_size] += top_diff[idxn, idxc, idxh, idxw] * self.weight[:, :, :, idxc]
        bottom_diff = bottom_diff[:, :, self.padding:self.padding +
                                  self.input.shape[2], self.padding:self.padding+self.input.shape[3]]
        return bottom_diff

    def forward_speedup(self, input):  # 优化后的前向传播
        self.input = input  # [N, C, H, W]
        height = self.input.shape[2] + self.padding * 2
        width = self.input.shape[3] + self.padding * 2
        self.input_pad = np.zeros(
            [self.input.shape[0], self.input.shape[1], height, width])
        self.input_pad[:, :, self.padding:self.padding+self.input.shape[2],
                       self.padding:self.padding+self.input.shape[3]] = self.input
        height_out = (height - self.kernel_size) // self.stride + 1
        width_out = (width - self.kernel_size) // self.stride + 1
        # input_col[N,H*W,C*K*K]
        self.input_col = np.zeros(
            [self.input.shape[0], height_out*width_out, self.input.shape[1]*(self.kernel_size**2)])
        for i in range(height_out):
            for j in range(width_out):
                self.input_col[:, i*width_out+j, :] = self.input_pad[:, :, i*self.stride:i*self.stride +
                                                                     self.kernel_size, j*self.stride:j*self.stride+self.kernel_size].reshape(self.input.shape[0], -1)
        # weight_col[C*K*K,Cout]
        weight_col = self.weight.reshape(-1, self.channel_out)
        # output_col[N*H*W,Cout]
        self.output_col = np.matmul(
            self.input_col.reshape(-1, self.input_col.shape[-1]), weight_col)
        # 恢复形状
        self.output = (self.output_col.reshape(
            self.input.shape[0], height_out, width_out, self.channel_out) + self.bias).transpose(0, 3, 1, 2)
        return self.output

    def backward_speedup(self, top_diff):  # 优化后的反向传播
        top_diff_reshaped = top_diff.transpose(
            0, 2, 3, 1).reshape(-1, self.channel_out)
        input_col_reshaped = self.input_col.reshape(
            -1, self.input_col.shape[-1])
        d_weight_col = np.matmul(input_col_reshaped.T, top_diff_reshaped)
        self.d_weight = d_weight_col.reshape(self.weight.shape)
        self.d_bias = np.sum(top_diff, axis=(0, 2, 3))
        bottom_diff = np.zeros(self.input_pad.shape)
        top_diff_pad = np.zeros([top_diff.shape[0], top_diff.shape[1], top_diff.shape[2] +
                                 2*self.kernel_size-2, top_diff.shape[3]+2*self.kernel_size-2])
        top_diff_pad[:, :, self.kernel_size-1:1-self.kernel_size,
                     self.kernel_size-1:1-self.kernel_size] = top_diff
        # top_diff_pad_col[N,H*W,C*K*K]
        top_diff_pad_col = np.zeros([self.input.shape[0], self.input_pad.shape[2]
                                     * self.input_pad.shape[3], top_diff.shape[1]*(self.kernel_size**2)])
        for i in range(self.input_pad.shape[2]):
            for j in range(self.input_pad.shape[3]):
                top_diff_pad_col[:, i*self.input_pad.shape[3]+j, :] = top_diff_pad[:, :, i*self.stride:i*self.stride +
                                                                                   self.kernel_size, j*self.stride:j*self.stride+self.kernel_size].reshape(self.input.shape[0], -1)
        weight_col = np.rot90(self.weight, k=2, axes=(1, 2)).transpose(
            3, 1, 2, 0).reshape(-1, self.channel_in)
        bottom_diff_col = np.matmul(
            top_diff_pad_col.reshape(-1, top_diff_pad_col.shape[-1]), weight_col)
        bottom_diff = bottom_diff_col.reshape(
            self.input.shape[0], self.input_pad.shape[2], self.input_pad.shape[3], self.channel_in).transpose(0, 3, 1, 2)
        bottom_diff = bottom_diff[:, :, self.padding:self.padding +
                                  self.input.shape[2], self.padding:self.padding+self.input.shape[3]]
        return bottom_diff

    def load_param(self, weight, bias):
        assert self.weight.shape == weight.shape
        assert self.bias.shape == bias.shape
        self.weight = weight
        self.bias = bias


class MaxPoolingLayer(object):
    def __init__(self, kernel_size, stride, type=0):
        self.kernel_size = kernel_size
        self.stride = stride
        self.forward = self.forward_raw
        self.backward = self.backward_raw
        if type == 1:  # 通过type选择加速版前向传播和反向传播函数
            self.forward = self.forward_speedup
            self.backward = self.backward_speedup
        print('\tMax pooling layer with kernel size %d, stride %d.' %
              (self.kernel_size, self.stride))

    def forward_raw(self, input):  # 优化前的前向传播
        self.input = input  # [N, C, H, W]
        self.max_index = np.zeros(self.input.shape)
        height_out = (self.input.shape[2] -
                      self.kernel_size) // self.stride + 1
        width_out = (self.input.shape[3] - self.kernel_size) // self.stride + 1
        self.output = np.zeros(
            [self.input.shape[0], self.input.shape[1], height_out, width_out])
        for idxn in range(self.input.shape[0]):
            for idxc in range(self.input.shape[1]):
                for idxh in range(height_out):
                    for idxw in range(width_out):
                        self.output[idxn, idxc, idxh, idxw] = np.max(
                            self.input[idxn, idxc, idxh * self.stride: idxh * self.stride + self.kernel_size, idxw * self.stride: idxw * self.stride + self.kernel_size])
                        curren_max_index = np.argmax(
                            self.input[idxn, idxc, idxh*self.stride:idxh*self.stride+self.kernel_size, idxw*self.stride:idxw*self.stride+self.kernel_size])
                        curren_max_index = np.unravel_index(
                            curren_max_index, [self.kernel_size, self.kernel_size])
                        self.max_index[idxn, idxc, idxh*self.stride +
                                       curren_max_index[0], idxw*self.stride+curren_max_index[1]] = 1
        return self.output

    def backward_raw(self, top_diff):  # 优化前的反向传播
        bottom_diff = np.zeros(self.input.shape)
        for idxn in range(top_diff.shape[0]):
            for idxc in range(top_diff.shape[1]):
                for idxh in range(top_diff.shape[2]):
                    for idxw in range(top_diff.shape[3]):
                        max_index = np.argmax(self.input[idxn, idxc, idxh*self.stride:idxh*self.stride +
                                              self.kernel_size, idxw*self.stride:idxw*self.stride+self.kernel_size])
                        max_index = np.unravel_index(
                            max_index, [self.kernel_size, self.kernel_size])
                        bottom_diff[idxn, idxc, idxh*self.stride+max_index[0], idxw *
                                    self.stride+max_index[1]] = top_diff[idxn, idxc, idxh, idxw]
        return bottom_diff

    def forward_speedup(self, input):  # 优化后的前向传播
        self.input = input
        height_out = (self.input.shape[2] -
                      self.kernel_size) // self.stride + 1
        width_out = (self.input.shape[3] - self.kernel_size) // self.stride + 1
        self.output = np.zeros(
            [self.input.shape[0], self.input.shape[1], height_out, width_out])
        self.input_col = np.zeros(
            [self.input.shape[0], self.input.shape[1], height_out*width_out, self.kernel_size**2])
        for i in range(height_out):
            for j in range(width_out):
                self.input_col[:, :, i*width_out+j, :] = self.input[:, :, i*self.stride:i*self.stride+self.kernel_size,
                                                                    j*self.stride:j*self.stride+self.kernel_size].reshape(self.input.shape[0], self.input.shape[1], -1)
        self.output_col = np.max(self.input_col, axis=3)
        max_index_col = np.zeros(
            [self.input.shape[0]*self.input.shape[1]*height_out*width_out, self.kernel_size**2])
        max_index_col[np.arange(self.input.shape[0]*self.input.shape[1]*height_out *
                                width_out), np.argmax(self.input_col, axis=3).reshape(-1)] = 1
        max_index_col = max_index_col.reshape(
            [self.input.shape[0], self.input.shape[1], height_out*width_out, self.kernel_size**2])
        self.output = self.output_col.reshape(
            [self.input.shape[0], self.input.shape[1], height_out, width_out])
        self.max_index = np.zeros(self.input.shape)
        for i in range(height_out):
            for j in range(width_out):
                self.max_index[:, :, i*self.stride:i*self.stride+self.kernel_size, j*self.stride:j*self.stride+self.kernel_size] = max_index_col[:,
                                                                                                                                                 :, i*width_out+j, :].reshape([self.input.shape[0], self.input.shape[1], self.kernel_size, self.kernel_size])
        return self.output

    def backward_speedup(self, top_diff):  # 优化后的反向传播
        bottom_diff = np.multiply(self.max_index, top_diff.repeat(
            self.kernel_size, 2).repeat(self.kernel_size, 3))
        return bottom_diff


class FlattenLayer(object):
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape
        assert np.prod(self.input_shape) == np.prod(self.output_shape)
        print('\tFlatten layer with input shape %s, output shape %s.' %
              (str(self.input_shape), str(self.output_shape)))

    def forward(self, input):
        assert list(input.shape[1:]) == list(self.input_shape)
        self.input = np.transpose(input, [0, 2, 3, 1])
        self.output = self.input.reshape(
            [self.input.shape[0]] + list(self.output_shape))
        return self.output

    def backward(self, top_diff):
        assert list(top_diff.shape[1:]) == list(self.output_shape)
        top_diff = np.transpose(top_diff, [0, 3, 1, 2])
        bottom_diff = top_diff.reshape(
            [top_diff.shape[0]] + list(self.input_shape))
        return bottom_diff
