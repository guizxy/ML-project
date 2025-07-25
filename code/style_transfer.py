from .layers1 import FullyConnectedLayer, ReLULayer, SoftmaxLossLayer
from .layers2 import ConvolutionalLayer, MaxPoolingLayer, FlattenLayer
from PIL import Image
import numpy as np
import imageio.v2 as imageio
import scipy.io
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def get_random_img(content_image, noise):
    noise_image = np.random.uniform(-20, 20, content_image.shape)
    random_img = noise_image * noise + content_image * (1 - noise)
    return random_img


class VGG19(object):
    def __init__(self, param_path):
        self.param_path = param_path
        self.param_layer_name = [
            'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
            'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
            'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
            'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
            'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4', 'pool5'
        ]

    def build_model(self):
        print('Building vgg-19 model...')

        self.layers = {}
        # 卷积核大小，输入通道数，输出通道数，填充大小，步长
        self.layers['conv1_1'] = ConvolutionalLayer(3, 3, 64, 1, 1, type=1)
        self.layers['relu1_1'] = ReLULayer()
        self.layers['conv1_2'] = ConvolutionalLayer(3, 64, 64, 1, 1, type=1)
        self.layers['relu1_2'] = ReLULayer()
        self.layers['pool1'] = MaxPoolingLayer(2, 2, type=1)

        self.layers['conv2_1'] = ConvolutionalLayer(3, 64, 128, 1, 1, type=1)
        self.layers['relu2_1'] = ReLULayer()
        self.layers['conv2_2'] = ConvolutionalLayer(3, 128, 128, 1, 1, type=1)
        self.layers['relu2_2'] = ReLULayer()
        self.layers['pool2'] = MaxPoolingLayer(2, 2, type=1)

        self.layers['conv3_1'] = ConvolutionalLayer(3, 128, 256, 1, 1, type=1)
        self.layers['relu3_1'] = ReLULayer()
        self.layers['conv3_2'] = ConvolutionalLayer(3, 256, 256, 1, 1, type=1)
        self.layers['relu3_2'] = ReLULayer()
        self.layers['conv3_3'] = ConvolutionalLayer(3, 256, 256, 1, 1, type=1)
        self.layers['relu3_3'] = ReLULayer()
        self.layers['conv3_4'] = ConvolutionalLayer(3, 256, 256, 1, 1, type=1)
        self.layers['relu3_4'] = ReLULayer()
        self.layers['pool3'] = MaxPoolingLayer(2, 2, type=1)

        self.layers['conv4_1'] = ConvolutionalLayer(3, 256, 512, 1, 1, type=1)
        self.layers['relu4_1'] = ReLULayer()
        self.layers['conv4_2'] = ConvolutionalLayer(3, 512, 512, 1, 1, type=1)
        self.layers['relu4_2'] = ReLULayer()
        self.layers['conv4_3'] = ConvolutionalLayer(3, 512, 512, 1, 1, type=1)
        self.layers['relu4_3'] = ReLULayer()
        self.layers['conv4_4'] = ConvolutionalLayer(3, 512, 512, 1, 1, type=1)
        self.layers['relu4_4'] = ReLULayer()
        self.layers['pool4'] = MaxPoolingLayer(2, 2, type=1)

        self.layers['conv5_1'] = ConvolutionalLayer(3, 512, 512, 1, 1, type=1)
        self.layers['relu5_1'] = ReLULayer()
        self.layers['conv5_2'] = ConvolutionalLayer(3, 512, 512, 1, 1, type=1)
        self.layers['relu5_2'] = ReLULayer()
        self.layers['conv5_3'] = ConvolutionalLayer(3, 512, 512, 1, 1, type=1)
        self.layers['relu5_3'] = ReLULayer()
        self.layers['conv5_4'] = ConvolutionalLayer(3, 512, 512, 1, 1, type=1)
        self.layers['relu5_4'] = ReLULayer()
        self.layers['pool5'] = MaxPoolingLayer(2, 2, type=1)

        self.layers['flatten'] = FlattenLayer([512, 7, 7], [25088])
        self.layers['fc6'] = FullyConnectedLayer(25088, 4096)
        self.layers['relu6'] = ReLULayer()
        self.layers['fc7'] = FullyConnectedLayer(4096, 4096)
        self.layers['relu7'] = ReLULayer()
        self.layers['fc8'] = FullyConnectedLayer(4096, 1000)
        self.layers['softmax'] = SoftmaxLossLayer()

        self.update_layer_list = []
        for layer_name in self.layers.keys():
            if 'conv' in layer_name:
                self.update_layer_list.append(layer_name)

    def init_model(self):
        print('Initializing parameters of each layer in vgg-19...')
        for layer_name in self.update_layer_list:
            self.layers[layer_name].init_param()

    def load_model(self):
        print('Loading parameters from file ' + self.param_path)
        params = scipy.io.loadmat(self.param_path)
        self.image_mean = params['normalization'][0][0][0]
        self.image_mean = np.mean(self.image_mean, axis=(0, 1))
        print('Get image mean: ' + str(self.image_mean))
        for idx in range(37):
            if 'conv' in self.param_layer_name[idx]:
                weight, bias = params['layers'][0][idx][0][0][0][0]
                weight = np.transpose(weight, [2, 0, 1, 3])
                bias = bias.reshape(-1)
                self.layers[self.param_layer_name[idx]
                            ].load_param(weight, bias)

    def load_image(self, image_dir, image_height, image_width):
        print('Loading and preprocessing image from ' + image_dir)
        # 如果是 RGBA 或灰度图，强制转换为 RGB
        image = imageio.imread(image_dir)
        if image.ndim == 2:  # 灰度图
            image = np.stack([image]*3, axis=-1)
        elif image.shape[2] == 4:  # RGBA图
            image = image[:, :, :3]
        self.input_image = image

        image_shape = self.input_image.shape
        self.input_image = np.array(Image.fromarray(
            self.input_image).resize((image_width, image_height)))
        self.input_image = np.array(self.input_image).astype(np.float32)
        self.input_image -= self.image_mean
        self.input_image = np.reshape(
            self.input_image, [1]+list(self.input_image.shape))
        # input dim [N, channel, height, width]
        self.input_image = np.transpose(self.input_image, (0, 3, 1, 2))
        return self.input_image, image_shape

    def save_image(self, input_image, image_shape, image_dir):
        print('Save image at ' + image_dir)
        # [N,C,H,W](0,1,2,3) -> [N,H,W,C](0,2,3,1)
        input_image = np.transpose(input_image, (0, 2, 3, 1))
        input_image = input_image[0] + self.image_mean
        input_image = np.clip(input_image, 0, 255).astype(np.uint8)
        imageio.imwrite(image_dir, input_image)

    def forward(self, input_image, layer_list):
        current = input_image
        layer_forward = {}
        for idx in range(len(self.param_layer_name)):
            current = self.layers[self.param_layer_name[idx]].forward(current)
            if self.param_layer_name[idx] in layer_list:
                layer_forward[self.param_layer_name[idx]] = current
        return layer_forward

    def backward(self, dloss, layer_name):
        layer_idx = list.index(self.param_layer_name, layer_name)
        for idx in range(layer_idx, -1, -1):
            dloss = self.layers[self.param_layer_name[idx]].backward(dloss)
        return dloss


class AdamOptimizer(object):
    def __init__(self, lr, diff_shape):
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8
        self.lr = lr
        self.mt = np.zeros(diff_shape)
        self.vt = np.zeros(diff_shape)
        self.step = 0

    def update(self, input, grad):
        self.step += 1
        self.mt = self.beta1*self.mt + (1-self.beta1)*grad
        self.vt = self.beta2*self.vt + (1-self.beta2)*(grad**2)
        mt_hat = self.mt / (1-self.beta1**self.step)
        vt_hat = self.vt / (1-self.beta2**self.step)
        output = input - self.lr * mt_hat / (np.sqrt(vt_hat + self.eps))
        return output
