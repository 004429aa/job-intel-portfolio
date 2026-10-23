# U3 照敲最小网络：单Linear层（没有隐藏层的"裸"网络）
# 规则：照敲，禁复制粘贴。跑通后改动测试（自己改一处，不问agent）

import torch
import torch.nn as nn  # nn=neural network，神经网络的各种零件都从这个工具包拿

# ========== 第一段：建网络 ==========
class MinNet(nn.Module):            # 自定义网络的标准写法：继承nn.Module这个基类，torch就知道它是网络了
    def __init__(self):             # 初始化：声明网络里有哪些零件
        super().__init__()          # 标准开头：先让父类初始化（照抄，别问，所有教程都这么写）
        self.fc = nn.Linear(784, 10)  # 核心就这一行。fc=fully connected（全连接层）
        self.fc2 = nn.Linear(10,10)
        # 784 = 28*28，把整张图拍扁成784个数塞进去（U2的[1,28,28]三维 → 拉直成一维）
        # 10 = 输出10个数，对应数字0-9各一个分数，谁分高就猜谁
        # Linear内部自己带两个参数：权重W(10行784列的矩阵)和偏置b(10个数)——训练要学的就是它们

    def forward(self, x):           # forward=前向：数据从进到出怎么流
        x = x.view(x.size(0), -1)   # 拍扁！[64,1,28,28]→[64,784]。view=重新排列不改数据
        # x.size(0)是这一批有几张图(64)，-1的意思是"剩下的维度自动算"(784)
        return self.fc(x)           # 过一遍全连接层，出[64,10]：每张图得到10个分数

model = MinNet()                    # 把网络类变成一个能用的实体（类=图纸，实例=造出来的机器）

# ========== 第二段：看懂网络里有什么 ==========
print("网络结构 =", model)                          # 会显示MinNet里有个fc: Linear(784->10)
print("参数总量 =", sum(p.numel() for p in model.parameters()))  # 数参数个数
#2个linear第一层784*10 + 10 = 7850个。7840个权重 + 10个偏置，第二层10*10＋10=110，总7960
w = model.fc.weight                 # 取出权重张量看看长啥样
print("权重形状 =", w.shape)                        # [10, 784]：10行784列
print("权重数值长啥样 =", w[0, :5])                 # 打印第0行的前5个——随机初始化的小数，训练就是调它们
print("权重梯度(训练前) =", w.grad)                 # None。还没训练，没算过梯度

# ========== 第三段：喂数据走一遍 ==========
from torchvision import datasets, transforms
train_set = datasets.MNIST(root="data", train=True, download=False, transform=transforms.ToTensor())

from torch.utils.data import DataLoader  # DataLoader=装卸工：把数据打包成一批一批(batch)
loader = DataLoader(train_set, batch_size=64)  # 每批64张图

images, labels = next(iter(loader))    # 取第一批。iter+next=从装卸工手里接过第一箱
# images1, labels1 = next(iter(loader))    # 取第二批。iter+next=从装卸工手里接过第一箱

print("一批图片形状 =", images.shape)   # [64, 1, 28, 28]：64张、1通道、28x28——四维！U1说过会见到
print("一批标签形状 =", labels.shape)   # [64]：64个答案

# print("二批图片形状 =", images1.shape)   # [64, 1, 28, 28]：64张、1通道、28x28——四维！U1说过会见到
# print("二批标签形状 =", labels1.shape)   # [64]：64个答案

outputs = model(images)                # 前向：一批图过网络
print("输出形状 =", outputs.shape)      # [64, 10]：每张图10个分数
print("第0张图的网络打分 =", outputs[0])  # 10个随机小数（还没训练，打的分是瞎打的）
print("瞎猜的结果 =", outputs[0].argmax().item(), "| 正确答案 =", labels[0].item())
# argmax=10个分数里挑最大的那个的位置。训练前≈扔骰子，训练后(U5)会越来越准

# ========== 验收打印 ==========
# 跑通后确认：参数7850、权重形状[10,784]、输出[64,10]、瞎猜的结果跟正确答案多半不一样
