import torch
from torch import nn

__all__ = ['AlexNet']

# 我们注意到AlexNet在初始化和后续使用的时候，都没有显式指定卷积核的初始值，这些就是我们要训练的模型参数
# 这是因为Pytorch支持模型参数和模型结构分离的策略，在定义模型结构时pytorch会自动初始化卷积核，同时也支持后续再模型定义完之后手动加载模型参数

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        # nn.sequential是允许用户按照特定的组合顺序组合多个计算层，形成一个完整的神经网络模型，从forward可以看到输入为图像，输出为特征图tensor
        self.conv = nn.Sequential(
            # nn.Conv2d是定义二维卷积的类，构造函数的参数是in_channels, out_channels, kernel_size, stride, padding
            nn.Conv2d(1, 96, 11, 4),
            # nn.ReLU返回一个激活函数类
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(3, 2)返回一个二维池化类，构造函数的参数是kernel_size, stride，kernel_size是kernel内进行的最大池化or平均池化
            nn.MaxPool2d(3, 2),
            # padding设为2来保证输入与输出的尺寸一致，且增大输出通道数
            nn.Conv2d(96, 256, 5, 1, 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
            # 连续3个卷积层，且使用更小的卷积窗口，除了最后的卷积层外，进一步增大了输出通道数
            nn.Conv2d(256, 384, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            # 最后进行一个池化层
            nn.MaxPool2d(3, 2)
        )
        # 利用nn.sequential构建全连接层，用于将特征图转为类别预测向量
        self.fc = nn.Sequential(
            # nn.Linear返回一个全连接类，构造函数的参数是输入神经元数，输出神经元数，是否偏置。可以看出FC层的卷积核数目为256x4096，就是不知道这里输入大小5x5是如何确定的
            nn.Linear(256 * 5 * 5, 4096),
            nn.ReLU(),
            # nn.Dropout返回一个Dropout类，训练时随机将输入张量的部分比例元素置0
            nn.Dropout(0.5),
            # 再次一个全连接层
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            # 再次一个全连接层，输出神经元为10，说明分类类别只有十类
            nn.Linear(4096, 10),
        )
    # forward函数定义了整个网络都前向传播途径，可以看到conv和fc的构造都放在了__init__函数里
    def forward(self, img):
        feature = self.conv(img)
        return self.fc(feature.view(img.shape[0], -1)) # 分成batch_size个全连接然后求和