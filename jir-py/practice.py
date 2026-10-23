# import pandas as pd
# # df = pd.read_excel(r"C:\Users\Administrator\Desktop\AI交叉岗位情报站.xlsx", sheet_name="岗位收集")
# print(df["城市"].value_counts())
# print(df["经验要求"].value_counts())
# print(df["薪资下限(k/月)"].mean(), "k 是薪资下限的平均值")
# from collections import Counter
# words = []
# for s in df["技能关键词"]:
#     s = str(s).replace("、", " ")
#     words += s.split()
# # print(Counter(words).most_common(10))
# #
# # word_count = Counter(words)
# # filter_words = [(word, num) for word, num in word_count.items() if num > 1]
# # print(sorted(filter_words, key=lambda x:x[1], reverse=True))
# print([(w, n) for w,n in Counter(words).most_common() if n>1])

a = ['1','2','3','4','5']
print('、'.join(a))