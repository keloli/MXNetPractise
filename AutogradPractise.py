# 参考资料：http://zh.gluon.ai/chapter_crashcourse/autograd.html
import mxnet.ndarray as nd
import mxnet.autograd as ag

x=nd.array(([1,2],[3,4]))
# print(x)
x.attach_grad() # 申请空间存放x的导数

# 定义f(x)=4*x*x
with ag.record():
	y=x*2
	z=y*x
# 通过z.backward()进行求导
z.backward()
print('x.grad:',x.grad)
# 验证求导结果是否为"4*x"
print(x.grad == 4*x)

# 对控制流求导
def f(a):
	b=a*2
	while nd.norm(b).asscalar()<1000:
		b=b*2
	if nd.sum(b).asscalar()>0:
		c=b
	else:
		c=100*b
	return c
# 像第一个例子一样，使用record记录，backward求导
a=nd.random_normal(shape=3)
a.attach_grad()
with ag.record():
	c=f(a)
c.backward()
print(a.grad==c/a)

# 头梯度和链式法则
with ag.record():
    y = x * 2
    z = y * x
# 头梯度相当于提供了在前面乘的系数
# 没有头梯度就默认系数为1
head_gradient = nd.array([[10, 1.], [.1, .01]])
z.backward(head_gradient)
print(x.grad)