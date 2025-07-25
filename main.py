from code.style_transfer import VGG19, AdamOptimizer, get_random_img
from code.layers2 import ConvolutionalLayer
from code.layers3 import ContentLossLayer, StyleLossLayer
import numpy as np
import time


def compute_MSE(data1, data2):
    errors = []
    for i in range(len(data1)):
        errors.append(data1[i] - data2[i])
    sqr_error = []
    for val in errors:
        sqr_error.append(pow(val, 2))
    return sum(sqr_error) / len(sqr_error)


def test_speedup():
    # 随机生成模拟数据
    test_data = np.random.rand(1, 256, 24, 40)  # 原始数据(N,C_in,H,W)
    test_dloss = np.random.rand(1, 256, 24, 40)  # 传入梯度(N,C_out,H_out,W_out)
    test_filter = np.random.rand(256, 3, 3, 256)  # 卷积核(C_in,K_h,K_w,C_out)
    test_bias = np.random.rand(256)  # 偏置
    # 测试
    conv = ConvolutionalLayer(3, 256, 256, 1, 1)
    conv.init_param()
    conv.load_param(test_filter, test_bias)
    stamp = time.time()
    conv_forward_result = conv.forward(test_data)
    conv_forward_time = time.time()-stamp
    print('conv forward raw time: %f ms' % (conv_forward_time*1000))
    stamp = time.time()
    conv_backward_result = conv.backward(test_dloss)
    conv_backward_time = time.time()-stamp
    print('conv backward raw time: %f ms' % (conv_backward_time*1000))

    speedup_conv = ConvolutionalLayer(3, 256, 256, 1, 1, 1)
    speedup_conv.init_param()
    speedup_conv.load_param(test_filter, test_bias)
    stamp = time.time()
    speedup_conv_forward_result = speedup_conv.forward(test_data)
    speedup_conv_forward_time = time.time()-stamp
    print('conv forward speedup time: %f ms' %
          (speedup_conv_forward_time*1000))
    stamp = time.time()
    speedup_conv_backward_result = speedup_conv.backward(test_dloss)
    speedup_conv_backward_time = time.time()-stamp
    print('conv backward speedup time: %f ms' %
          (speedup_conv_backward_time*1000))

    speedup_conv_forward_mse = compute_MSE(
        conv_forward_result.flatten(), speedup_conv_forward_result.flatten())
    speedup_conv_backward_mse = compute_MSE(
        conv_backward_result.flatten(), speedup_conv_backward_result.flatten())
    if speedup_conv_forward_mse < 0.003 and speedup_conv_backward_mse < 0.003:
        print('SPEEDUP CONV TEST PASS.')
    else:
        print('SPEEDUP CONV TEST FAILED.')
        exit()

    print('CONV FORWARD SPEEDUP RATIO: %f' %
          (conv_forward_time / speedup_conv_forward_time))
    print('CONV BACKWARD SPEEDUP RATIO: %f' %
          (conv_backward_time / speedup_conv_backward_time))


if __name__ == '__main__':
    np.random.seed(5813)
    print('-------------------------')
    test_speedup()
    print('-------------------------')
    CONTENT_LOSS_LAYERS = ['relu4_2']
    STYLE_LOSS_LAYERS = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']
    NOISE = 0.2
    ALPHA, BETA = 20, 100
    TRAIN_STEP = 300
    LEARNING_RATE = 1.0
    IMAGE_HEIGHT, IMAGE_WIDTH = 512, 768
    # 模型初始化
    vgg = VGG19(param_path='./src/vgg19.mat')
    vgg.build_model()
    vgg.init_model()
    vgg.load_model()
    content_loss_layer = ContentLossLayer()
    style_loss_layer = StyleLossLayer()
    adam_optimizer = AdamOptimizer(1.0, [1, 3, IMAGE_HEIGHT, IMAGE_WIDTH])
    content_image, content_shape = vgg.load_image(
        './src/content/venice.jpg', IMAGE_HEIGHT, IMAGE_WIDTH)
    style_image, _ = vgg.load_image(
        './src/style/style.jpg', IMAGE_HEIGHT, IMAGE_WIDTH)
    content_layers = vgg.forward(content_image, CONTENT_LOSS_LAYERS)
    style_layers = vgg.forward(style_image, STYLE_LOSS_LAYERS)
    transfer_image = get_random_img(content_image, NOISE)
    # 前向传播与反向传播
    start = time.time()
    for step in range(TRAIN_STEP):
        transfer_layers = vgg.forward(
            transfer_image, CONTENT_LOSS_LAYERS + STYLE_LOSS_LAYERS)
        content_loss = np.array([])
        style_loss = np.array([])
        content_diff = np.zeros(transfer_image.shape)
        style_diff = np.zeros(transfer_image.shape)
        for layer in CONTENT_LOSS_LAYERS:
            # 计算内容损失的前向传播
            current_loss = content_loss_layer.forward(
                transfer_layers[layer], content_layers[layer])
            content_loss = np.append(content_loss, current_loss)
            # 计算内容损失的反向传播
            dloss = content_loss_layer.backward(
                transfer_layers[layer], content_layers[layer])
            content_diff += vgg.backward(dloss, layer)
        for layer in STYLE_LOSS_LAYERS:
            # 计算风格损失的前向传播
            current_loss = style_loss_layer.forward(
                transfer_layers[layer], style_layers[layer])
            style_loss = np.append(style_loss, current_loss)
            # 计算风格损失的反向传播
            dloss = style_loss_layer.backward(
                transfer_layers[layer], style_layers[layer])
            style_diff += vgg.backward(dloss, layer)
        total_loss = ALPHA * np.mean(content_loss) + BETA * np.mean(style_loss)
        image_diff = ALPHA * content_diff / \
            len(CONTENT_LOSS_LAYERS) + BETA * \
            style_diff / len(STYLE_LOSS_LAYERS)
        # 利用Adam优化器对风格迁移图像进行更新
        transfer_image = adam_optimizer.update(transfer_image, image_diff)
        if step % 5 == 0:
            print('Step %d, loss = %f' %
                  (step, total_loss), content_loss, style_loss)
            print('cost time: %f' % (time.time() - start))
            vgg.save_image(transfer_image, content_shape,
                           './output/output_' + str(step) + '.jpg')
            start = time.time()
