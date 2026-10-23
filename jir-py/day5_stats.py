import pandas as pd#导入pandas：整列整列地算数据（openpyxl是镊子逐格夹，pandas是铲车整列铲）
df = pd.read_excel(r"E:\Aaaaaaa.Item\Item--Job Information Report\AI交叉岗位情报站.xlsx", sheet_name="岗位收集")#一整页表直接读成一个df（DataFrame），30行12列的完整数据都在里面
print(df["城市"].value_counts())#df["城市"]拿到城市这一整列，value_counts()一口气数完各城市出现次数——铲车式操作，一行顶openpyxl一个循环
print(df["经验要求"].value_counts())#同上，数经验要求的分布（结果：不限23条占77%，说明市场对入门者友好）
print(df["薪资下限(k/月)"].mean(), "k 是薪资下限的平均值")#mean()求这列数字的平均值：30条岗位的薪资下限平均15.6k
from collections import Counter#导入Counter：专门数东西个数的工具，给它一个列表它还你一份"词→次数"的账（从collection这个工具包里面只拿出counter这个工具）
words = []
for s in df["技能关键词"]:#逐行过技能关键词这一列，s每次是一行（如"Python、深度学习、C/C++"）
    s = str(s).replace("、", " ")#清洗：30条数据里顿号和空格混用（18条顿号/7条空格/4条混用），先把顿号统一换成空格，不然"Python"和"Python、"会被数成两个不同的词
    words += s.split()#split()按空格把一行切成一个词列表，+=把30行的词全汇进words这个大池子
print(Counter(words).most_common(10))#数池子里每个词的频次，most_common(10)取前10——这份Top10榜单就是全项目的核心情报：Python 23次断层第一
print(df["学历要求"])#打印学历整列原始值（调试用：看真实数据长什么样，能看到"在读研究生/在校应届硕士"等4种口语化写法——脏数据用眼睛确认过了才好清洗）
word =[]
for a in df["学历要求"]:#和上面技能词频同一个套路，换了个统计对象
    a = str(a).replace("、", " ")#同样的清洗：统一分隔符
    word += a.split()#同样的汇池子
print([(w, n) for w,n in Counter(word).most_common() if n>1])#列表推导：数完频次只留下出现次数>1的词——过滤掉只出现1次的杂音（4种"在读"写法各只出现1次，被滤掉后榜单只剩本科20/硕士及以上5，口径自然归一）