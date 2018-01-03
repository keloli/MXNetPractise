# _*_ coding: utf-8 _*_
# http://zh.gluon.ai/chapter_crashcourse/ndarray.html

from mxnet import ndarray as nd

x1=nd.zeros((3,4))
print("x1是:%s"%x1)

x2 = nd.ones((3, 4))
print("x2是:%s"%x2)

x3=nd.array([[1,2],[2,1]])
print("x3是:%s"%x3)

x4=nd.random_normal(0,1,shape=(3,4))
print("x3是:%s \n x4是:%s"%(x3,x4))

print(x4.shape)
print(x4.size)

x5=nd.dot(x1,x2.T)
print(x5)

# 广播
# 当二元操作符左右两边ndarray形状不一样时，
# 系统会尝试将其复制到一个共同的形状。
# 例如a的第0维是3, b的第0维是1，那么a+b时会将b沿着第0维复制3遍：
a = nd.arange(3).reshape((3,1))
b = nd.arange(2).reshape((1,2))
print('a:', a)
print('b:', b)
print('a+b:', a+b)

c1=nd.arange(4)
print(c1)
c2=nd.arange(4).reshape((2,2))
print(c2)

# ndarray可以很方便地同numpy进行转换
import numpy as np
x = np.ones((2,3))
y = nd.array(x)  # numpy -> mxnet
z = y.asnumpy()  # mxnet -> numpy
print([z, y])

# 替换操作
x=nd.ones((3,4))
y=nd.ones((3,4))
before=id(y)
y=x+y
c=(id(y)==before)
print(c)

z=nd.zeros_like(x)
print(z)
before=id(z)
z[:]=x+y
print(id(z)==before)

# 如果要避免上面x,y的临时开销，可以使用操作符的全名版本中的out参数
nd.elemwise_add(x,y,out=z)
print(id(z)==before)

# 截取
x=nd.arange(0,9).reshape((3,3))
print("x:是%s" %x)
print(x[0:2])

x[1,2]=9.0
print(x)

# 多维截取
x=nd.arange(0,9).reshape((3,3))
print('x:',x)
print(x[1:2,1:3]) 
# 通过上面两个截取实验可以看出，x的行和列从0开始编号，
# 1:2表示从第1行（从第0行开始计数）截取到第（2-1）行
# 1:3表示从第1列（从第0列开始计数）截取到第（3-1）列

# ndarray模块提供一系列多维数组操作函数。所有函数列表可以参见：
# https://mxnet.incubator.apache.org/api/python/ndarray.html



