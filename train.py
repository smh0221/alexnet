import os
import time
import torch
import warnings
from data import *
from model import *


def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        # 如果没指定device就使用net的device
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            net.eval() # 评估模式, 这会关闭dropout
            acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
            net.train() # 改回训练模式
            n += y.shape[0]
    return acc_sum / n


def train(net, train_loader, test_loader, optimizer, device, epoch_num):
    # 将模型部署在cpu或cuda上
    net = net.to(device)
    print("training on ", device)
    # 定义损失函数，采用了预定义的交叉熵
    loss = torch.nn.CrossEntropyLoss()
    # loop epoch
    for epoch in range(epoch_num):
        train_l_sum, train_acc_sum, n, batch_idx, start = 0.0, 0.0, 0, 0, time.time()
        # train_loader可以用迭代器的方式访问，具体的返回取决于DataSet的定义，返回的数据量应该是一个batch的
        for X, y in train_loader:
            print("batch_idx: ", batch_idx)
            # 数据部署到设备上
            X = X.to(device)
            y = y.to(device)
            # 模型的前向传播。数据送入网络，自动调用的是网络的forward函数，实际是在nn.Module的成员函数__call__中调用的forward函数
            y_hat = net(X)
            # 损失函数的前向传播。调用loss的__call__成员函数
            l = loss(y_hat, y)
            # 每个batch优化前进行梯度清零
            optimizer.zero_grad()
            # 损失函数的反向传播，此时损失函数是拿到了计算图的，因此反向传播也是针对整个模型的
            l.backward()
            # 根据反向传播的梯度更新参数
            optimizer.step()
            # 损失值累加
            train_l_sum += l.cpu().item()
            # 准确度累加
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            # 累加batch_size
            n += y.shape[0]
            # 做完一个batch
            batch_idx += 1
        test_acc = evaluate_accuracy(test_loader, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_idx, train_acc_sum / n, test_acc, time.time() - start))


def get_args_parser(add_help=True):
    import argparse
    parser = argparse.ArgumentParser(description="PyTorch AlexNet Training", add_help=add_help)
    parser.add_argument("--data-path", default="/home/shangminghao/dataset/fashion_mnist", type=str, help="dataset path")
    parser.add_argument("-d", "--device", default="cuda", type=str, help="cpu or cuda (defaule: cuda)")
    parser.add_argument("-b", "--batch-size", default=128, type=int, help="batch size for training (default: 128)")
    parser.add_argument("-e", "--epochs", default=5, type=int, metavar="N", help="epochs for training (default: 5)")
    parser.add_argument(
        "-j", "--workers", default=4, type=int, metavar="N", help="number of data loading workers (default: 4)"
    )
    parser.add_argument("-o", "--opt", default="adam", type=str, help="optimizer (defaule: adam)")
    parser.add_argument("-l", "--lr", default=0.001, type=float, help="initial learning rate (default: 0.001)")
    parser.add_argument("--output-path", default="/home/shangminghao/workspace/alexnet/train_output", type=str, help="train output path")
    return parser


def main(args):
    if args.output_path and not os.path.exists(args.output_path):
        os.mkdir(args.output_path)
    device = torch.device(args.device) # 初始化设备类型
    train_loader, test_loader = create_data_loader(args.data_path, args.batch_size, args.workers) # 初始化数据集
    net = AlexNet() # 初始化模型
    if args.opt == "sgd":
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr) # 初始化优化器
    else:
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr) # 初始化优化器
    train(net, train_loader, test_loader, optimizer, device, args.epochs) # 开始训练
    torch.save(net, f"{args.output_path}/alexnet.pth") # 保存训练结果
    print(net) # 打印模型


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
