# U4 照敲训练循环四拍：forward → loss → backward → step（整个项目的心脏）
# 规则：照敲，禁复制粘贴。跑通后改动测试（自己改一处，不问agent）
# 本单元不追求训练完整——四拍各敲一遍、看懂打印，就算过关。完整训练是U5

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ========== 准备（U3的老熟人，快速过） ==========
class MinNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(784, 10)  # 这次先用单层，四拍看清楚再谈复杂
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc(x)

model = MinNet()
train_set = datasets.MNIST(root="data", train=True, download=False, transform=transforms.ToTensor())
loader = DataLoader(train_set, batch_size=32)

# ========== 训练三件套（第四拍前必须先有这三样） ==========
criterion = nn.CrossEntropyLoss()                                  # 裁判：算"猜错了多少"。交叉熵损失，分类任务标配,MSEloss做分类，标签和输出要同形状。
# CrossEntropyLoss可以直接接收数字标签不用转one-hot，所以分类任务不能随便替换
optimizer = torch.optim.SGD(model.parameters(), lr=1.0)            # 调参手：负责改小数。SGD=随机梯度下降；lr=学习率=每次改的步子大小
images, labels = next(iter(loader))                                # 拿一批64张

snapshot = model.fc.weight.detach().clone()  # 整个权重拍快照存起来，等下对比。clone()=真复制一份独立内存

# ========== 四拍来了（就这7行，全项目核心） ==========
# 第一拍 forward：算猜答案
outputs = model(images)          # 64张图过网络 → [64,10] 打分

# 第二拍 loss：算错多少
loss = criterion(outputs, labels)  # 打分对比正确答案 → 一个数（误差）。数值越大错得越离谱
print("loss =", loss.item())

# 第三拍 backward：算每个参数该往哪边改
loss.backward()                  # 反向传播：从loss出发倒着算，给7850个参数每个算出"梯度"=往哪个方向调、调多急

# 第四拍 step：真的去改
optimizer.step()                 # 调参手按梯度动工，7850个小数每个都微调一下。lr=1.0决定步子多大

# ========== 验收对比 ==========
# 坑：不能随便挑几个权重看——MNIST图片边缘像素几乎全黑(=0)，黑像素对应的权重梯度恰好是0
# (梯度公式≈误差×像素，像素0则梯度0)。要看就挑"梯度最大的那个位置"，保证看得见变化
grad = model.fc.weight.grad                      # backward之后grad从None变成了有内容的张量
pos = grad.abs().argmax()                        # 全局找梯度绝对值最大的那个位置(下标)
row, col = divmod(pos.item(), 784)               # 位置换算回第几行第几列。divmod=除法同时拿商和余数
print(f"梯度最大的位置: 第{row}行第{col}列, 梯度值 = {grad[row, col].item():.6f}")
before_w = snapshot[row, col]                    # 从快照里取同一位置的训练前权重
print(f"该位置训练前权重 = {before_w:.6f}")
print(f"该位置训练后权重 = {model.fc.weight[row, col].item():.6f}")  # 变了！四拍真的改了小数
print(f"验证: 变化量应≈ -lr×梯度 = {-1.0 * grad[row, col].item():.6f}")  # SGD规则: 新值=旧值-lr×梯度

# ========== 留给U5的问题（想想，不用写代码） ==========
# 现在只训了一批。要训完6万张（约938批），四拍外面套个循环——这就是U5
