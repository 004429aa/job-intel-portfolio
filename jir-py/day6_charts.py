import pandas as pd#导入pandas：整列整列地读数据
import matplotlib.pyplot as plt#导入matplotlib的画图模块，起别名plt——全行业通用写法，少打十几个字母
from collections import Counter#导入计数器：数每个技能出现几次

plt.rcParams["font.sans-serif"] = ["SimHei"]#全局设置字体为黑体SimHei。不设这行，matplotlib默认字体里没有中文，所有中文标签都会变成方框——这是它最著名的坑
# plt.rcParams["axes.unicode_minus"] = False#配套设置：黑体里没有负号字形，不关掉它负号也变方框（unicode_minus=用unicode负号）

df = pd.read_excel(r"E:\Aaaaaaa.Item\Item--Job Information Report\AI交叉岗位情报站.xlsx", sheet_name="岗位收集")#读整页表进df

# 图1：技能词频Top10 条形图（我自改成竖向bar，标签遮挡自己解决的）
words = []
for s in df["技能关键词"]:#逐行过技能关键词列，汇词池（和D5同一个套路）
    words += str(s).replace("、", " ").split()#先统一分隔符再切词，清洗
top10 = Counter(words).most_common(10)#数频次取前10，此刻是"从大到小"排
print(top10)
top10.reverse()#反转成从小到大。因为bar是从左往右画——不反转的话Python(最大)画在最左，反转后画在最右，形成阶梯登顶的视觉
names = [w for w, n in top10]#拆出技能名列表（w=name, n=count的解包写法）
counts = [n for w, n in top10]#拆出对应次数列表，两条平行列表按位置对应
plt.figure(figsize=(8.0, 10.0))#新建一张画布，宽8英寸高10英寸（英寸！不是英尺），竖版——给下面10个旋转的中文标签留够地方
plt.bar(names, counts, color="steelblue")#竖向柱状图：names做x轴、counts做柱高（bar=竖，barh=横，中文长标签横排会互相遮挡，这是我把barh改成bar后撞上的问题）
plt.title("30条AI交叉岗位 · 技能需求Top10")#图标题
plt.ylabel("出现次数")#y轴标签（竖向图数值在y轴）
plt.tight_layout()#自动微调边距，防止标题轴标签被裁掉
plt.xticks(rotation=30, ha='right')#x轴标签旋转30度、右对齐——10个中文技能名不旋转会叠在一起（自己试出来的解法）
plt.subplots_adjust(left=0.1, bottom=0.12)#手动再留边：底部12%空间给旋转后的标签，不然长词底部出框
# plt.savefig("技能Top10.png", dpi=150)#存成PNG文件，dpi=150够打印清晰度。图从此是独立文件，报告直接引用
plt.show()#弹窗展示。注意：必须先savefig再show——show会把画布清空，顺序反了存出来是空白图

# 图2：薪资分布直方图
df["薪资中值"] = (df["薪资下限(k/月)"] + df["薪资上限(k/月)"]) / 2#现场造一列新数据：每条岗位薪资上下限取平均（原表没这列，派生列）
plt.figure(figsize=(8, 5))#新画布，横版
plt.hist(df["薪资中值"], bins=10, color="steelblue", edgecolor="white")#直方图：bins=10把薪资范围切成10个区间，每个区间数有几条岗位落进去、画一根柱。edgecolor白边让柱子之间有分界（我改过bins=8→10，颗粒度看数据量定）
plt.title("30条岗位 · 薪资中值分布")
plt.xlabel("薪资中值(k/月)")#x轴标签
plt.ylabel("岗位数")#y轴标签
plt.tight_layout()
# plt.savefig("薪资分布.png", dpi=150)
plt.show()
