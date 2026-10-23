# U9 loss曲线 + 预测样例可视化（给训练过程"装仪表盘"）
# 规则：照敲，禁复制粘贴。跑通后改动测试（自己改一处，不问agent）
# 25分钟单元：照敲15min + 改动测试5min + 说人话5min
# CPU训练3轮约5分钟，画图秒出

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# D6老坑配置：中文标题不配SimHei会变方块，负号不关unicode_minus会显示不出来
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ========== 选手：U7的SimpleCNN原样搬来（98.8%那台） ==========
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(16, 64, kernel_size=5, padding=2)
        self.fc = nn.Linear(64 * 7 * 7, 10)
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))   # [64,16,14,14]
        x = self.pool(torch.relu(self.conv2(x)))   # [64,64,7,7]
        x = x.view(x.size(0), -1)                  # 拍扁→[64,3136]
        return self.fc(x)

model = SimpleCNN()
train_set = datasets.MNIST(root="data", train=True, download=False, transform=transforms.ToTensor())
test_set = datasets.MNIST(root="data", train=False, download=False, transform=transforms.ToTensor())
test_loader = DataLoader(test_set, batch_size=1000)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

def evaluate(model, test_loader):
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            preds = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
    return correct / 10000

# ========== 新东西1：训练时把每轮loss记到列表里 ==========
EPOCHS = 5
loss_history = []                                   # 仪表盘的数据来源：每轮的平均loss存这里
acc_history = []
for epoch in range(EPOCHS):
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    total_loss = 0
    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / 938  # 938批，除完是这轮的平均loss
    acc =evaluate(model, test_loader)
    loss_history.append(avg_loss)                   # U5只打印不留档，U9留档才能画曲线
    acc_history.append(acc)
    print(f"第{epoch+1}轮: loss={avg_loss:.4f} 准确率={evaluate(model, test_loader)*100:.1f}%")

# ========== 新东西2：画loss曲线 ==========
# plt.figure(figsize=(8, 5))
# plt.plot(range(1, EPOCHS + 1), loss_history, marker='o', color='red')
# plt.xlabel('训练轮数')
# plt.ylabel('平均loss')
# plt.title(f'U9 loss曲线（{EPOCHS}轮，lr=0.1）')
# plt.grid(True)
# # plt.savefig('u9_loss曲线.png', dpi=150)
# plt.show()

# ========== 新东西3：预测样例可视化（对错标色） ==========
sample_loader = DataLoader(test_set, batch_size=16)          # 从测试集舀16张
sample_images, sample_labels = next(iter(sample_loader))
with torch.no_grad():
    sample_preds = model(sample_images).argmax(dim=1)       # 只猜不学，同evaluate

fig, axes = plt.subplots(4, 4, figsize=(10, 10))
for i, ax in enumerate(axes.flat):
    ax.imshow(sample_images[i][0], cmap='gray')             # [0]取第1通道，灰度图只有一个通道
    right = sample_preds[i].item() == sample_labels[i].item()
    ax.set_title(f"预测{sample_preds[i].item()} 实际{sample_labels[i].item()}",
                 color='green' if right else 'red')          # 对=绿 错=红，一眼看出翻车没
    ax.axis('off')
plt.suptitle('U9 预测样例（绿=对 红=错）')
plt.tight_layout()
# plt.savefig('u9_预测样例.png', dpi=150)
plt.show()

fig, ax1 = plt.subplots(figsize=(8, 5))

# 左y轴：loss（红色）
ax1.plot(range(1, EPOCHS + 1), loss_history, marker='o', color='red', label='平均Loss')
ax1.set_xlabel('训练轮数')
ax1.set_ylabel('平均loss', color='red')
ax1.tick_params(axis='y', labelcolor='red')
ax1.grid(True)

# 右y轴：准确率（蓝色）
ax2 = ax1.twinx()
ax2.plot(range(1, EPOCHS + 1), acc_history, marker='s', color='blue', label='测试准确率')
ax2.set_ylabel('准确率', color='blue')
ax2.tick_params(axis='y', labelcolor='blue')

plt.title(f'U9 loss+准确率曲线（{EPOCHS}轮，lr=0.1）')
fig.legend(loc="upper right")
plt.tight_layout()
# plt.savefig('u9_loss_acc曲线.png', dpi=150)
plt.show()

# ========== 验收打印（跑完自查） ==========
# 1. loss曲线应该是往下掉的——loss越小说明模型和答案贴得越近
# 2. 8张样例大概率全绿（准确率98%+，8张全错的概率极低）
# 3. 两张PNG落在Jir-DL目录：u9_loss曲线.png、u9_预测样例.png

# ========== 改动测试候选（自己挑一处改，跑完看变化） ==========
# A. EPOCHS改成5：曲线多两个点，loss应掉得更低
# B. 挑样例改成16张：subplots(4, 4)，axes.flat自动适配
# C. loss曲线旁边再画一条准确率曲线：每轮也存acc_history，两条线画在同一张图
