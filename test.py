import torch
import torchvision

from PIL import Image


def main(args):
    net = torch.load(args.model_path) # 加载模型
    net = net.to(args.device) # 部署模型
    net.eval() # 把模型转为推理模式，推理模式下模型的运行会有所不同，例如dropout不会随意丢弃神经元
    img = Image.open(args.data_path) # 读取代预测图像
    # 导入图片，利用torchvision处理图片
    trans = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Grayscale(1),
        ])
    img = trans(img)
    img = img.to(args.device)
    img = img.unsqueeze(0) # 图片扩展多一维,因为输入到保存的模型中是4维的[batch_size, 通道, 长, 宽]，而普通图片只有三维，[通道, 长, 宽]
    # Fashion Minist数据对应这十个类别
    classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
    output = net(img)
    # torch.nn.functional模块和torch.nn.Module模块不同，它集中了许多函数操作可以直接调用，不需要创建层的方式调用
    prob = torch.nn.functional.softmax(output, dim=1) # softmax是将某个维度的张量归一化到(0,1)，且整个维度的和为1
    print("概率：", prob)
    value, predicted = torch.max(output.data, dim=1) # 找到分数最大的值和对应的索引
    predict = output.argmax(dim=1)
    pred_class = classes[predicted.item()] # 由于predicted是张量，所以用item得到标量
    print("预测类别：", pred_class)


def get_args_parser(add_help=True):
    import argparse
    parser = argparse.ArgumentParser(description="PyTorch AlexNet Training", add_help=add_help)
    parser.add_argument("--model-path", default="/home/shangminghao/workspace/alexnet/train_output/alexnet.pth", type=str, help="model params path")
    parser.add_argument("--data-path", default="./bag.jpg", type=str, help="test data path")
    parser.add_argument("-d", "--device", default="cuda", type=str, help="cpu or cuda (defaule: cuda)")
    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)