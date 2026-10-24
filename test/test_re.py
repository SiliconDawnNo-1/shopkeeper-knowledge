"""
    该案例演示了正则常用方法
    正则对应的模块是re
    使用正则方式1:  调用多次，对于相同的匹配规则，需要写多次parttern
        re.search(r"","")
        re.match(r"","")
        re.sub()
        re.findall()
        re.finditer()
    使用正则方式2：   用多次，对于相同的匹配规则，只需要编译一次就行
        # 先编译
        rep = re.compile(parttern)
        # 用编译后的正则对象调用相关的方法
        rep.search    全文搜索，返回第一个匹配的match对象
        rep.match     从头开始进行匹配
        rep.sub("要替换的内容","查找的文本")       正则替换
        rep.findall   全文搜索，返回所有匹配的内容列表，里面放的是字符串
        rep.finditer  全文搜索，返回所有匹配的内容列表，里面放的是match对象
"""
import re

md_content = "这里有一张图 ![示意图](images/demo(1).png) 再来一张 ![图片](images/demo(1).png)"
image_name = "demo(1).png"
# 编译正则（不变） 正则中有两类字符 元字符 . * ? ! [a-z] {} 字面字符 a  png
rep = re.compile(r"\!\[.*?\]\(.*?" + re.escape(image_name) + r".*?\)")

# print(rep.sub("![你好](aaa/bbb.jpg)", md_content))

# 1. search：全文找【第一个】匹配 → 返回 match 对象
# match_obj = rep.search(md_content)
# print("1. search 结果：", match_obj.group() if match_obj else "未找到")
# print(match_obj.group(0))
# print(f"start:{match_obj.start()}")
# print(f"end:{match_obj.end()}")
# print(f"span:{match_obj.span()}")

# 2. match：只从【开头】匹配 → 这里开头不是图片语法，必找不到
# match_first = rep.match(md_content)
# print("2. match 结果：", "找到" if match_first else "未找到（只匹配开头）")

# 3. findall：找【所有】匹配 → 返回字符串列表
# all_list = rep.findall(md_content)
# print("3. findall 列表：", all_list)

# 4. finditer：找【所有】匹配 → 返回 match 对象（你要的“search 匹配所有”）
# print("4. finditer 遍历所有：")
# for m in rep.finditer(md_content):
#     print("  - 内容：", m.group(), " | 位置：", m.span())

# () 和 ?  贪婪匹配
text = "A123B456B"
# 1. 用 .*  → 贪婪（吃到最后）
# print(re.findall(r"A(.*)B", text))  # 输出：['123B456']
# 2. 用 .*? → 非贪婪（找到第一个就停）
print(re.findall(r"A(.*?)B", text)) # 输出：['123']

