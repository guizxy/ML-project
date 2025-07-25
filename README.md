# 机器学习课程作业-项目介绍
## 基于 NumPy 的风格迁移实现
这是一个使用Python和NumPy从零开始实现的神经风格迁移项目，其核心思想源于Leon A.Gatys等人的论文"A Neural Algorithm of Artistic Style"。项目不依赖任何深度学习框架(如TensorFlow或PyTorch)，所有网络层、损失函数和优化器均为**手动**实现。

该项目能够将一张“**内容图**”的结构与一张“**风格图**”的艺术风格相结合，生成一张全新的、融合了两者特点的图像。

## 主要特性
**纯NumPy实现**：所有核心算法，包括卷积层、池化层、损失函数等，都只使用 NumPy库实现。
**VGG19 模型**: 使用了预训练的 VGG19 模型作为特征提取器来感知图像的内容和风格。
**内容与风格损失**：
- ***内容***损失通过计算特征图之间的**均方误差**(MSE)来定义。
- ***风格***损失通过计算特征图的**格拉姆矩阵**(Gram Matrix)之间的均方误差来定义。
- 优化实现: 卷积层和池化层同时包含了**基础**的循环实现(raw)和使用im2col思想优化的speedup版本，用于对比学习和性能评估。

**Adam 优化器**：项目内置了一个手动实现的Adam优化器，用于在训练过程中迭代更新生成的图像。

文件结构
```
.
├── code/
│   ├── __init__.py             
│   ├── layers1.py              # 定义全连接层和ReLU激活层
│   ├── layers2.py              # 定义卷积层和最大池化层(包含优化版本)
│   ├── layers3.py              # 定义内容损失层和风格损失层
│   ├── style_transfer.py       # 定义了VGG19模型的结构、Adam优化器
├── output/                     # 存储代码的输出的图片
├── src/                        # 存储代码运行中需要用到的资源
│   ├── content/                # 存储内容图片
│   ├── style/                  # 存储风格图片
│   │   ├── style.jpg           # 风格图片-梵高的《星空》
│   ├── vgg19.mat               # VGG19预训练权重
├── main.py                     # 项目主入口，用于运行测试和风格迁移
```
## 环境配置
1. 克隆或下载项目
2. 安装依赖库
    项目主要依赖 numpy, scipy, Pillow 和 imageio。可以通过 pip 安装：
    ```Bash
    pip install numpy scipy pillow imageio
    ```
3. 准备图像
准备您自己的内容图和风格图，并根据 main.py 中的路径设置将它们放置在正确的目录中。
4. 如何运行
- 确保所有依赖、模型和图像文件都已准备就绪。
- 在项目根目录下，运行 main.py 文件：
    ```Bash
    python main.py
    ```
- 程序将首先运行 test_speedup 函数，打印优化前后卷积层的**性能对比**。
- 接着，程序将开始风格迁移的迭代过程，每隔一定步数，会在 output/ 目录下生成一张当前的风格迁移图像，如 output_0.jpg, output_5.jpg...。
## 配置与自定义
您可以直接在 main.py 文件的开头部分修改超参数以调整风格迁移的效果：
- CONTENT_LOSS_LAYERS, STYLE_LOSS_LAYERS: 修改用于计算损失的VGG层。
- ALPHA, BETA: 调整内容损失和风格损失的权重，这是影响最终效果的关键参数。BETA 越大，风格化效果越强。
- TRAIN_STEP: 训练迭代的总步数。
- LEARNING_RATE: Adam 优化器的学习率。
- IMAGE_HEIGHT, IMAGE_WIDTH: 生成图像的尺寸。
- 也可以在代码中修改内容图和风格图的文件路径。