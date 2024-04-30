import torch
import torchvision

__all__ = ['create_data_loader']

# 加载数据集
def create_data_loader(root, batch_size, workers):
    trans = []
    trans.append(torchvision.transforms.Resize(size=224)) # 添加Resize对象到列表中
    trans.append(torchvision.transforms.ToTensor()) # 添加ToTensor对象到列表中用于将PIL图像转为numpy矩阵
    transform = torchvision.transforms.Compose(trans) # 用列表初始化Compose对象
    mnist_train = torchvision.datasets.FashionMNIST(root=root, train=True, download=True, transform=transform) # 返回一个数据集实例，用于训练
    mnist_test = torchvision.datasets.FashionMNIST(root=root, train=False, download=True, transform=transform) # 返回一个数据集实例，用于验证
    # 创建了一个数据加载器，从mnist_train数据集中加载和打乱数据，并分为batch_size大小的批次，用于后续的神经网络训练
    train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=workers)
    test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=workers)

    return train_loader, test_loader