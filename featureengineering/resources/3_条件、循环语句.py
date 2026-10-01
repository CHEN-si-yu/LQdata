# -*- coding: utf-8 -*-

"""
程序开头注释
author: xingbuxing
date: 2017年04月16日
功能：本程序主要介绍python的条件语句、循环语句
"""


# =====条件语句介绍
# 条件语句语法如下：
"""
if 条件A（结果为布尔值，true或者False）:
    执行相关操作1（需要使用tab缩进）
    ......

elif 条件B（结果为布尔值，true或者False）:
    执行相关操作2
    ......

else:
    执行相关操作3
"""

# 条件语句解释说明如下：
"""
1. 若条件A为True，那么执行相关操作1，程序结束
2. 若条件A为False，那么判断条件B，若条件B为True，那么执行相关操作2，程序结束
3. 若条件A为False，那么判断条件B，若条件B为False，那么执行相关操作3，程序结束
"""

# 条件语句示例：根据股票代码，判断股票来自于哪个市场
# stock_code = 'sh600000'  # 尝试将stock_code改成'sz000002'，'aapl'看相关结果。
#
# if stock_code.startswith('sh'):
#     print '上海股票'
# elif stock_code.startswith('sz'):
#     print '深圳股票'
# else:
#     print '不知道哪里来的股票'


# =====for循环语句介绍
# for循环是最常用的循环语句
# 案例1：计算1+2+3+……+10
# sum_num = 0  # 用于存储计算的结果
# for i in range(10 + 1):
#     sum_num += i  # 此处需要使用tab按键进行缩进
#     print i, sum_num

# 案例2：批量判断股票来自于哪个市场
# stock_code_list = ['sh600000', 'sh600001', 'sz000001', 'aapl']
# for stock_code in stock_code_list:
#
#     # 若stock_code是sh600000
#     if stock_code == 'sh600000':
#         print stock_code, '此股票为浦发银行，我已经知道它是来自上海的股票'
#         continue  # 跳过此次循环，不运行接下来的语句，直接进入下个循环
#         # break  # 停止整个循环，跳出for语句
#
#     # 判断股票来自于哪个市场
#     if stock_code.startswith('sh'):
#         print stock_code, '上海股票'
#     elif stock_code.startswith('sz'):
#         print stock_code, '深圳股票'
#     else:
#         print stock_code, '不知道哪里来的股票'


# =====while语句
# while语句语法如下：
"""
while 条件A:
    执行相关操作1（需要使用tab缩进）
    ......
"""

# 条件语句解释说明如下：
"""
1. 判断条件A，若条件A为False，那么程序结束。
2. 判断条件A，若条件A为True，那么执行相关操作1。
3. 然后再次判断条件A，重复上面的步骤
"""

# while语句案例1：计算1+2+3+……+10
num = 1
max_num = 10
sum_num = 0  # 存储计算结果
while num <= max_num:
    sum_num += num
    num += 1
    print sum_num

# while语句案例2：计算1+2+3+……+10
num = 1
max_num = 10
sum_num = 0
while True:
    sum_num += num
    num += 1
    print sum_num, num
    if num == max_num+1:
        break
