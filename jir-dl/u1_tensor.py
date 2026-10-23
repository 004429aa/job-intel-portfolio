# U1 张量第一操：建 / 查 / 算

import torch  # 导入PyTorch，行业惯例直接叫torch。它管的核心数据结构叫"张量(Tensor)"

# ========== 第一段：建 ==========
a = torch.tensor([[1.0, 2.0], [3.0, 4.0],[5.0, 6.0]])  # 用现成的数据建张量。三维的，可以理解成一张表格（3行2列）
b = torch.zeros(3)          # 建3个0：一行三列的零张量，占位用
c = torch.rand(2, 3)        # 建2行3列的随机数（0~1之间）。神经网络初始权重就是这玩意生成的
print("a =", a)
print("b =", b)
print("c =", c)

print("a的形状 shape =", a.shape)     # shape=形状，(2,2)意思是2行2列。查形状是调bug第一动作
print("a的数据类型 dtype =", a.dtype)  # dtype=data type。建的时候写了1.0带小数点，所以是float32
print("a的第0行 =", a[0])             # 中括号取行，从0开始数（不是从1）
print("a的第0行第1列的元素 =", a[0, 1])  # 逗号隔开行列，取到单个数。结果应该是2

x = torch.tensor([10.0, 20.0, 30.0])
y = torch.tensor([1.0, 2.0, 3.0])
print("x + y =", x + y)        # 对应位置相加：[11, 22, 33]
print("x * 2 =", x * 2)        # 每个位置乘2：[20, 40, 60]
print("x的均值 =", x.mean().item())    # 全部加起来除以个数：20。训练里的loss就是靠mean算出来的
