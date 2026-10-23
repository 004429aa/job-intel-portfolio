# U2 加载MNIST + matplotlib看数据：深度学习"Hello World"
from torchvision import datasets  # torchvision是torch的视觉工具包，datasets里存着现成的公开数据集
from torchvision import transforms  # transforms=变换：数据进模型前要做的预处理都从这拿

# ========== 第一段：下载并加载 ==========
# MNIST：6万张手写数字图（0-9），28x28像素，灰度。深度学习最经典的入门数据集
# transform：把图片变成张量（ToTensor()）。原图是PIL图片格式，torch不认，必须先转
train_set = datasets.MNIST(
    root="data",              # 存到哪个文件夹（相对路径，会在脚本旁边建data文件夹）
    train=True,               # True=训练集6万张，False=测试集1万张
    download=True,            # 本地没有就下载（第一次跑会慢，下完以后秒开）
    transform=transforms.ToTensor()  # 图片→张量，顺手把像素值从0-255压到0-1
)

print("训练集图片数 =", len(train_set))          # 应该是60000
img, label = train_set[0]                        # 取第0条数据。train_set[i]给两样东西：图和标签，没有shuffle都是固定的
print("图片形状 =", img.shape)                    # [1, 28, 28]：1通道（灰度）x 28行x 28列——三维张量！
print("像素值范围 =", img.min().item(), "到", img.max().item())  # 0到1，ToTensor压过了
print("这张图的标签 label =", label)              # 0-9里的某个数，模型要学的就是看图猜这个数

# ========== 第二段：用matplotlib看图（复用D6技能） ==========
import matplotlib.pyplot as plt  # D6的老朋友，画图流程跟当时一样

plt.rcParams["font.sans-serif"] = ["SimHei"]  # D6踩过的坑：不设黑体，中文全变方框
plt.rcParams["axes.unicode_minus"] = False    # 配套：负号也不变方框

fig, axes = plt.subplots(5, 2, figsize=(10, 5))  # subplots：一次画多张小图。5行2列共10个格子
for i in range(100,110):                              # 取10条数据，画进10个格子
    image, lab = train_set[i-100]
    ax = axes[(i-100) % 5, (i-100)// 5]                     # 定位格子：i除以5得行数(i=0-4在第0行)，除5余数得列
    ax.imshow(image.squeeze(0), cmap="viridis")     # imshow=显示图片。squeeze(0)把[1,28,28]压成[28,28]（imshow不要通道那维）；cmap=viridis渐变色显示显示
    ax.set_title(f"标签: {lab}")                  # 每张图标上正确答案
    ax.axis("off")                               # 关掉坐标轴刻度，看图更干净
plt.tight_layout()                               #自动微调整边距
# plt.savefig("mnist_样例.png", dpi=150)  # 先存后show（D6的坑：show会清空画布，顺序反了存出空白图）
plt.show()

# ========== 验收打印 ==========
# 跑通后确认：60000张、形状[1,28,28]、像素0到1、弹窗显示10张手写数字图
