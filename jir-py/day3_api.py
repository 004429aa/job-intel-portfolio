import requests#导入requests模块：专门负责跟网上服务器收发数据的工具包
url = "https://api.deepseek.com/chat/completions"#快递收件地址：DeepSeek服务器的接口
headers = {"Authorization": "Bearer sk-这里换成你的新key（去平台重新生成）"}#通行证，验明身份用的（密钥跟着这行走）
data = {"model": "deepseek-chat",
        "messages": [{"role": "user", "content": "用一句话解释什么AI交叉岗"}]}#快递包裹：用哪个模型+问它什么话
resp = requests.post(url, headers=headers, json=data)#把地址、通行证、包裹一起寄出，回信装进resp（这行会停几秒，在等服务器回信）
print(resp.status_code)#查看快递签收状态，200=成功
print(resp.json())#1. 自动把接口返回的字节内容拿出来，转成字符串2. 内部自动调用了 json.loads ()，返回 python 字典。（这不是json用法，是requests的）
print(resp.json()["choices"][0]["message"]["content"])#拆信封：回信是一大坨JSON，这串下标一路捅到"它回答的正文"那一层
