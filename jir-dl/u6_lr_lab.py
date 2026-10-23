# U6 改动实验室：学习率与轮数实验，建立超参数手感
# 规则：模板只用跑，实验记录自己写。这次反过来——代码照敲一次，然后自己设计实验
# 核心问题：lr太大/太小/刚好，训练曲线各长什么样？

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class MinNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(784, 10)
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc(x)

train_set = datasets.MNIST(root="data", train=True, download=False, transform=transforms.ToTensor())
test_set = datasets.MNIST(root="data", train=False, download=False, transform=transforms.ToTensor())
test_loader = DataLoader(test_set, batch_size=1000)

def evaluate(model, test_loader):
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
    return correct / 10000

# ========== 实验函数：一次实验=一组lr+轮数，返回每轮准确率 ==========
def run_experiment(lr, epochs):
    model = MinNet()                                   # 每次实验重建模型——权重要从随机重新开始才公平
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    accs = []                                          # 存每一轮的考试准确率
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
        acc = evaluate(model, test_loader)
        accs.append(acc)
        print(f"  第{epoch+1}轮: loss={total_loss/938:.4f} 准确率={acc*100:.1f}%")
    return accs

# ========== 实验一：lr=1.0 太大（步子太大会怎样？） ==========
print("实验一: lr=1.0 (故意太大)")
run_experiment(lr=1.0, epochs=3)

# ========== 实验二：lr=0.001 太小（步子太小会怎样？） ==========
print("实验二: lr=0.001 (故意太小)")
run_experiment(lr=0.001, epochs=3)

# ========== 实验三：lr=0.1 基准（U5的老朋友） ==========
print("实验三: lr=0.1 (基准)")
run_experiment(lr=0.1, epochs=3)

# ========== 你的实验记录写在这里（改动测试就是它） ==========
# 在Jir-DL下新建 u6_实验记录.md，把三次实验的每轮准确率抄进去，然后回答：
# 1. lr=1.0 的准确率曲线什么形状？（升/降/震荡/不升？）
'''第1轮: loss=0.4142 准确率=91.2%
  第2轮: loss=0.3333 准确率=90.2%
  第3轮: loss=0.3213 准确率=90.4%，震荡'''

# 2. lr=0.001 的曲线什么形状？
'''
  第1轮: loss=1.9248 准确率=75.8%
  第2轮: loss=1.3999 准确率=81.1%
  第3轮: loss=1.1267 准确率=82.9%,上升
'''
# 3. 用一句大白话总结"学习率"是什么
