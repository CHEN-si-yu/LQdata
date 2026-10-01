# -*- coding: utf-8 -*-

"""
此为多行注释方式
author: xingbuxing
date: 2017年04月16日
功能：本程序主要介绍python的常用的数据类型，以及计算符号。希望以后大家只要看这个程序，就能回想起相关的基础知识。
"""

# ===注释的用法
# 在每一行的开头，加上#，是对该行进行单行注释
# print 'hello world'  # 行末注释，在一句程序的末尾，一般用来解释这句话。注意空格。

# control + / 多行同时注释或取消注释（mac上是command + /）。
# 尝试同时取消注释或注释下面三行代码
# print 'hello world'
# print 'hello world'
# print 'hello world'


# ===数字,num
# stock_num = 3000  # 整数，有3000只股票在交易
# print stock_num, type(stock_num)  # type()函数的作用是输出变量的类型
# stock_price = 19.88  # 浮点数，股票价格是19.88
# print stock_price, type(stock_price)
# stock_price = .08  # 小数点之前的0可以省略，.08和0.08是一样的。港股中有很多的仙股，它的价格只有几分钱
# print stock_price, type(stock_price)
# stock_pe = -27  # 负数的表示方式，股票市盈率-27
# print stock_pe, type(stock_pe)
# stock_value = 5.23E10  # 可以使用科学技术发来表示很大的数字，股票市值
# print stock_value, type(stock_value)


# ===字符串,string
# stock_name = '浦发银行'
# print stock_name, type(stock_name)
# stock_code = "sh600000"
# print stock_code, type(stock_code)


# ===布尔值：只有两个，True和False。大小写敏感
# print True, False, type(True)


# ===空值：只有一个，None。大小写敏感。表示没有值的值
# print None, type(None)


# ===变量的名称，一定要有意义，
# 不要使用a、b、c、aa等无意义的变量名
# 取名规则：首字母需要是字母或下划线，其余部分可以是字母，下划线和数字


# ===算术符号, + - * / %
# 以加法为例子，可以把下面的加号变成- * /其他符号。
# old_stock_num = 3000  # 现有股票数量3000只
# new_stock_num = 600  # 新增股票600只
# all_stock_num = old_stock_num + new_stock_num  # 全部股票共有多少只？
# print all_stock_num

# % 取余数的操作
# print 9 % 3

# 同类型变量进行运算，结果还是该类型变量
# print 5 / 3  # 整数除以整数，结果是整数1，而不是1.666666...
# 小数和整数进行运算，结果是小数
# print 5.0 / 3  # 小数除以整数，结果是浮点数，结果是1.666666...

# 自运算的快速写法
# stock_num = 3000
# stock_num += 600  # 效果等同于：stock_num = stock_num + 600。可以把加号变成- * /其他符号。
# print stock_num


# ===比较运算> < >= <= == !=
# num1 = 10
# num2 = 20
# print num1 > num2  # 判断num1是否大于num2，输出结果是布尔变量
# print num1 >= num2  # 判断num1是否大于等于num2
# print num1 == num2  # 判断num1是否等于num2
# print num1 != num2  # 判断num1是否不等于于num2


# ===布尔运算and or & |
# and &，两者都为真，才是真
# print 2 > 1 and 2 != 1  # 两者都是True，输出结果就是True
# print 2 > 1 and 2 == 1  # 其中有一个为False，输出结果就是False

# or | 至少一个为真，就是真
# print 2 > 1 or 2 == 1  # 其中有一个为True，输出结果就是True
