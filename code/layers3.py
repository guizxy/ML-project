import numpy as np


class ContentLossLayer(object):  # 内容损失层
    def __init__(self):
        print('\tContent loss layer.')

    def forward(self, input_layer, content_layer):  # 内容损失层前向传播
        # L = ∑(input - content)^2 / (2 * NCHW)
        loss = (np.sum(input_layer.reshape(-1) - content_layer.reshape(-1))
                ** 2) * 0.5 / content_layer.size
        return loss

    def backward(self, input_layer, content_layer):  # 内容损失层反向传播
        # ▽L = (input - content) / NCHW
        bottom_diff = (input_layer - content_layer) / content_layer.size
        return bottom_diff


class StyleLossLayer(object):  # 风格损失层
    def __init__(self):
        print('\tStyle loss layer.')

    def forward(self, input_layer, style_layer):  # 风格损失层前向传播(计算Gram矩阵)
        # 计算A--gram_style
        # style_layer[N,C,H,W] -> style_layer_reshape[N,C,H * W]
        style_layer_reshape = np.reshape(
            style_layer, [style_layer.shape[0], style_layer.shape[1], -1])
        # A(n,i,j) = ∑ Y(n,i,h,w)Y(n,j,h,w) 目标风格图像的风格特征A
        self.gram_style = np.array([np.matmul(
            style_layer_reshape[n, :, :], style_layer_reshape[n, :, :].T) for n in range(style_layer.shape[0])])

        # 计算G--gram_input
        # input_layer[N,C,H,W] -> input_layer_reshape[N,C,H * W]
        self.input_layer_reshape = np.reshape(
            input_layer, [input_layer.shape[0], input_layer.shape[1], -1])
        # G(n,i,j) = ∑ X(n,i,h,w)X(n,j,h,w) 风格迁移图像的风格特征G
        self.gram_input = np.array([np.matmul(
            self.input_layer_reshape[n, :, :], self.input_layer_reshape[n, :, :].T) for n in range(input_layer.shape[0])])

        # 计算第I层风格损失
        M = input_layer.shape[2] * input_layer.shape[3]  # W * H
        N = input_layer.shape[1]  # C
        self.div = M * M * N * N  # W^2 * H^2 * C ^ 2
        # G - A
        style_diff = self.gram_input - self.gram_style
        # (∑(G - A)^2) / (4 * N * C^2 * H^2 * W^2)
        loss = np.sum(style_diff**2) / (self.div * style_layer.shape[0] * 4.0)
        return loss

    def backward(self, input_layer, style_layer):  # 风格损失层反向传播
        # ->bottom_diff(N,C,H*W)
        bottom_diff = np.zeros(
            [input_layer.shape[0], input_layer.shape[1], input_layer.shape[2]*input_layer.shape[3]])
        for n in range(input_layer.shape[0]):
            # ▽L = (∑X(G-A)) / (N * C^2 * H^2 * W^2)
            bottom_diff[n, :, :] = np.matmul(
                (self.gram_input[n, :, :] - self.gram_style[n, :, :]).T, self.input_layer_reshape[n, :, :]) / (self.div * style_layer.shape[0])
        bottom_diff = np.reshape(bottom_diff, input_layer.shape)
        return bottom_diff
