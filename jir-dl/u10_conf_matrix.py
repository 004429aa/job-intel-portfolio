# U10 工业风数据集：NEU热轧带钢表面缺陷6分类（从MNIST迈向真实工业场景）
# 数据已备好：data_neu/{train,test}/{6类}/ 各240/60张，200x200灰度钢面图
# 规则：照敲，禁复制粘贴。跑通后改动测试（自己改一处，不问agent）
# CPU训练：1440张/批64=23批每轮，比MNIST的938批快得多，5轮几分钟搞定

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ========== 和MNIST的三个不同（照敲时注意） ==========
# 1. 图片是PNG文件不是内置数据集 → 用ImageFolder按文件夹名自动打标签
# 2. PNG载入是3通道(自动复制成RGB) → Grayscale()转回1通道（本来就是灰度钢面图）
# 3. 尺寸200x200太大 → Resize到128x128。两次池化后32x32，fc输入量可控
transform = transforms.Compose([
    transforms.Grayscale(),          # 3通道→1通道
    transforms.Resize((128,128)),     # 200x200→128x128
    transforms.ToTensor(),           # 转张量+归一化到0~1
])

train_set = datasets.ImageFolder(root="data_neu/train", transform=transform)
test_set = datasets.ImageFolder(root="data_neu/test", transform=transform)
print("类别映射:", train_set.class_to_idx)
# 疯狂输出类别和数字的对应：0:crazing裂纹 1:inclusion夹杂 2:patches斑块
# 3:pitted_surface点蚀 4:rolled_in_scale氧化皮 5:scratches划痕
test_loader = DataLoader(test_set, batch_size=60)

# ========== 选手：U9的SimpleCNN改三处 ==========
class DefectCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=2)   # 改1：输入还是1通道(灰度化后)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(16, 64, kernel_size=5, padding=2)
        self.fc = nn.Linear(64 * 32 * 32, 6)                      # 改2：128x128两次池化→32x32；改3：输出6类
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))   # [B,16,64,64]
        x = self.pool(torch.relu(self.conv2(x)))   # [B,64,32,32]
        x = x.view(x.size(0), -1)                  # 拍扁→[B,65536]
        return self.fc(x)

model = DefectCNN()
print("参数量:", sum(p.numel() for p in model.parameters()))

criterion = nn.CrossEntropyLoss()
# 为什么不用SGD lr=0.1？——已替你踩过坑：SGD 0.1跑5轮只有25%~36%（震荡学不动）。
# 工业纹理特征梯度大且难学，Adam(自适应步长) lr=0.001 是这类任务的常规起点
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def evaluate(model, test_loader):
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            preds = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
    return correct / len(test_set)

# ========== 训练：U9同款（loss+准确率都留档） ==========
EPOCHS = 15
loss_history = []
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
    avg_loss = total_loss / len(train_loader)
    acc = evaluate(model, test_loader)
    loss_history.append(avg_loss)
    acc_history.append(acc)
    print(f"第{epoch+1}轮: loss={avg_loss:.4f} 准确率={acc*100:.1f}%")

# ========== 双轴曲线（U9同款） ==========
plt.figure(figsize=(8, 5))
ax1 = plt.gca()
ax1.plot(range(1, EPOCHS + 1), loss_history, marker='o', color='red', label='平均Loss')
ax1.set_xlabel('训练轮数'); ax1.set_ylabel('平均loss', color='red')
ax1.tick_params(axis='y', labelcolor='red'); ax1.grid(True)
ax2 = ax1.twinx()
ax2.plot(range(1, EPOCHS + 1), acc_history, marker='s', color='blue', label='测试准确率')
ax2.set_ylabel('准确率', color='blue'); ax2.tick_params(axis='y', labelcolor='blue')
plt.title(f'U10 NEU缺陷分类（{EPOCHS}轮，Adam lr=0.001）')
# plt.savefig('u10_NEU曲线1.png', dpi=150)
plt.show()

# ========== 预测样例：16张对错标色（U9同款） ==========
sample_loader = DataLoader(test_set, batch_size=16, shuffle=True)   # shuffle：每类都可能抽到
sample_images, sample_labels = next(iter(sample_loader))
with torch.no_grad():
    sample_preds = model(sample_images).argmax(dim=1)

CLASSES = ['裂纹', '夹杂', '斑块', '点蚀', '氧化皮', '划痕']   # class_to_idx顺序对应
fig, axes = plt.subplots(4, 4, figsize=(11, 11))
for i, ax in enumerate(axes.flat):
    ax.imshow(sample_images[i][0], cmap='gray')
    right = sample_preds[i].item() == sample_labels[i].item()
    ax.set_title(f"预测{CLASSES[sample_preds[i]]} 实际{CLASSES[sample_labels[i]]}",
                 color='green' if right else 'red', fontsize=9)
    ax.axis('off')
plt.suptitle('U10 NEU缺陷预测样例（绿=对 红=错）')
plt.tight_layout()
# plt.savefig('u10_预测样例1.png', dpi=150)
plt.show()

num_classes = 6
conf_matrix = np.zeros((num_classes, num_classes), dtype=int)

model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        preds = outputs.argmax(dim=1)
        # 逐批填充混淆矩阵：真实i，预测j
        for i, j in zip(labels.cpu().numpy(), preds.cpu().numpy()):
            conf_matrix[i][j] += 1

# 打印混淆矩阵
print("\n===== 6×6 混淆矩阵（行=真实，列=预测）=====")
print(conf_matrix)

print("\n===== 每类召回率 & 最容易错分目标 =====")
for i in range(num_classes):
    row_sum = conf_matrix[i].sum()
    tp = conf_matrix[i, i]
    recall = tp / row_sum if row_sum > 0 else 0

    # 把对角线置0，找剩下里面最大值的索引（错得最多）
    row_copy = conf_matrix[i].copy()
    row_copy[i] = 0
    worst_j = np.argmax(row_copy)
    worst_cnt = row_copy[worst_j]

    print(f"{CLASSES[i]} | 召回率：{recall:.3f} | 最容易错分成：{CLASSES[worst_j]}（错{worst_cnt}个）")

# 【可选：画混淆矩阵热力图，面试展示更好看】
plt.figure(figsize=(7,6))
plt.imshow(conf_matrix, cmap="Blues")
plt.colorbar()
plt.xticks(range(6), CLASSES, rotation=45)
plt.yticks(range(6), CLASSES)
plt.xlabel("预测类别")
plt.ylabel("真实类别")
plt.title("NEU钢缺陷 混淆矩阵")
# 在格子里面写数字
for ii in range(6):
    for jj in range(6):
        plt.text(jj, ii, str(conf_matrix[ii][jj]), ha="center", va="center")
plt.tight_layout()
plt.savefig("u10_confusion_matrix.png", dpi=150)
plt.show()

