# U7 两层CNN vs 单层Linear：正面交锋（深度学习"深"字兑现的地方）
# 规则：照敲，禁复制粘贴。跑通后改动测试（自己改一处，不问agent）
# CPU训练一轮约1-2分钟，共3轮+对比组，全程约5分钟

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ========== 双方选手 ==========
class LinearNet(nn.Module):              # 选手A：U1-U6的老朋友，纯Linear
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(784, 10)
    def forward(self, x):
        x = x.view(x.size(0), -1)       # 拍扁[64,1,28,28]→[64,784]，形状信息全丢
        return self.fc(x)

class SimpleCNN(nn.Module):              # 选手B：两层卷积+池化，新选手
    def __init__(self):
        super().__init__()
        # 卷积层：小滤镜在图上滑动扫特征。in=1通道(灰度)，out=16个不同滤镜
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, padding=2)
        # padding=2：图边圈补一圈0，保证5x5滤镜扫完后图大小不变(28x28→28x28)
        self.pool = nn.MaxPool2d(kernel_size=2)   # 池化：2x2格子里只留最大值，图缩半(28→14)。作用=保留最强信号+减计算量
        self.conv2 = nn.Conv2d(16, 64, kernel_size=5, padding=2)  # 第二层卷积：16种初级特征→64种高级特征组合，
        # 实验对比过：out_channels 16→64 准确率小幅上涨；kernel 3→5 参数量大幅增加（同通道下，5×5 核的权重是 3×3 的 25/9≈2.78 倍）。
        # 【注】conv1实际是1→16通道，本行"32→64"是相对默认写法而言的改动描述，不是conv1的形状，别混。
        self.fc = nn.Linear(64 * 7 * 7, 10)       # 最后还是要Linear打分。7=28经两次池化(28→14→7)
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # 卷积→relu(负数归零,留下兴奋点)→池化 [64,16,14,14]
        # x = torch.relu(self.conv1(x))#全连接层维度不匹配
        x = self.pool(torch.relu(self.conv2(x)))  # 再来一轮 [64,64,7,7]
        x = x.view(x.size(0), -1)                 # 拍扁[64,64,7,7]→[64,3136]
        return self.fc(x)                         # [64,10]打分

# ========== 裁判和场地（老熟人） ==========
train_set = datasets.MNIST(root="data", train=True, download=False, transform=transforms.ToTensor())
test_set = datasets.MNIST(root="data", train=False, download=False, transform=transforms.ToTensor())
test_loader = DataLoader(test_set, batch_size=1000)
criterion = nn.CrossEntropyLoss()

def evaluate(model, test_loader):
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            preds = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
    return correct / 10000

def train(model, epochs=3, lr=0.1):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    for epoch in range(epochs):
        train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
        total_loss = 0
        for images, labels in train_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"  第{epoch+1}轮: loss={total_loss/938:.4f} 准确率={evaluate(model, test_loader)*100:.1f}%")

# ========== 比赛 ==========
print("选手A LinearNet（U5-U6基准，参数7850个）:")
a = LinearNet()
print("参数量:", sum(p.numel() for p in a.parameters()))
train(a)

print("选手B SimpleCNN（参数量见打印）:")
b = SimpleCNN()
print("参数量:", sum(p.numel() for p in b.parameters()))
train(b)

# ========== 验收打印 ==========
# 跑通后看两件事：
# 1. 两个网络的参数量各多少？（CNN参数量可能比Linear还少，但准确率反而高——为什么？）
# 2. 最终准确率：Linear约92%，CNN应该97%+。差距就是"看形状"与"看像素"的差距
