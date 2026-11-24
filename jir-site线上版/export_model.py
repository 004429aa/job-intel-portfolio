from sentence_transformers import SentenceTransformer

# 1. 在线加载模型（利用你已有的缓存）
model = SentenceTransformer("BAAI/bge-small-zh-v1.5")

# 2. 直接保存到线上版的 model 文件夹
# 这个 save 方法会自动把软链接解引用，拷出真实的 .safetensors 权重，并且自动适配你当前的库版本！
model.save("model")