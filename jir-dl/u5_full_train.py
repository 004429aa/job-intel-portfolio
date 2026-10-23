# U5 完整训练：四拍外面套循环，看准确率爬升（前四个单元的总兑现）
# 规则：照敲，禁复制粘贴。跑通后改动测试（自己改一处，不问agent）
# CPU训练一轮约1-2分钟，跑的时候别关窗口

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ========== 准备（全是老熟人） ==========
class MinNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(784, 10)
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc(x)

model = MinNet()
train_set = datasets.MNIST(root="data", train=True, download=False, transform=transforms.ToTensor())
test_set = datasets.MNIST(root="data", train=False, download=False, transform=transforms.ToTensor())
# 注意train=False：测试集1万张。训练时不给它看，训完用它考试——检查是背会了还是真会了
test_loader = DataLoader(test_set, batch_size=1000)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)  # lr回到0.1。U4的1.0是实验，正式训练步子太大会震荡，lr小准确率会变低

# ========== 考试函数：在测试集上算准确率 ==========
def evaluate(model, test_loader):
    correct = 0                                        # 答对的人数
    with torch.no_grad():                              # 考试不用学习！no_grad=关掉梯度记录，省内存算得快
        for images, labels in test_loader:             # 1万张分10批，每批1000
            outputs = model(images)                    # 只要第一拍：猜答案（考试不算loss不更新参数）
            preds = outputs.argmax(dim=1)              # 每行10个分里挑最大位置。dim=1=沿着第1维(列)找
            correct += (preds == labels).sum().item()  # 猜的跟答案对比，对一条记一条
    return correct / 10000                             # 对的除以总数 = 准确率

print(f"训练前考试: 准确率 = {evaluate(model, test_loader)*100:.1f}%")  # 应该≈10%（瞎猜10选1）

# ========== 训练循环：四拍外面套两层 ==========
EPOCHS = 5                                             # epoch=把6万张全部过一遍叫"一轮"过的轮数多会增加准确率
for epoch in range(EPOCHS):
    train_loader = DataLoader(train_set, batch_size=64, shuffle=False)  # shuffle=每轮洗牌，防网络背顺序
    total_loss = 0
    for images, labels in train_loader:                # 938批，每批64张
        # ---------- 四拍（U4照敲过的那7行，一字不改） ----------
        outputs = model(images)                        # 一拍 forward
        loss = criterion(outputs, labels)              # 二拍 loss
        optimizer.zero_grad()                          # 关键新增！清空上一批的旧梯度，不清会累加(越加越错)，梯度会越大，更新方向错误
        loss.backward()                                # 三拍 backward
        optimizer.step()                               # 四拍 step
        # ------------------------------------------------------
        total_loss += loss.item()                      # 累计这轮的总误差

    acc = evaluate(model, test_loader)                 # 每轮结束考一次试
    print(f"第{epoch+1}轮: 平均loss = {total_loss/938:.4f}, 考试准确率 = {acc*100:.1f}%")

# ========== 验收打印 ==========
# 跑通后确认：训练前≈10%，第1轮结束后≈90%上下，第3轮更高——前四个单元埋的种子全部兑现
