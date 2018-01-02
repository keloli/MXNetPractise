#代码来源：http://zh.gluon.ai/chapter_supervised-learning/linear-regression-scratch.html
from mxnet import ndarray as nd
from mxnet import autograd

num_inputs = 2
num_examples = 1000

true_w = [2, -3.4]
true_b = 4.2

X = nd.random_normal(shape=(num_examples, num_inputs))
y = true_w[0] * X[:, 0] + true_w[1] * X[:, 1] + true_b
print(X[0], y[0])
y += .01 * nd.random_normal(shape=y.shape)

import random
batch_size = 10
def data_iter():
    # 产生一个随机索引
    idx = list(range(num_examples))
    random.shuffle(idx)
    for i in range(0, num_examples, batch_size):
        j = nd.array(idx[i:min(i+batch_size,num_examples)])
        yield nd.take(X, j), nd.take(y, j)

for data, label in data_iter():
    print(data, label)
    break


w = nd.random_normal(shape=(num_inputs, 1))
b = nd.zeros((1,))
params = [w, b]


for param in params:
    param.attach_grad()

def net(X):
    return nd.dot(X, w) + b

def square_loss(yhat, y):
    # 注意这里我们把y变形成yhat的形状来避免自动广播
    return (yhat - y.reshape(yhat.shape)) ** 2

def SGD(params, lr):
    for param in params:
        param[:] = param - lr * param.grad


epochs = 7
learning_rate = .001
for e in range(epochs):
    total_loss = 0
    for data, label in data_iter():
        with autograd.record():
            output = net(data)
            loss = square_loss(output, label)
        loss.backward()
        SGD(params, learning_rate)

        total_loss += nd.sum(loss).asscalar()
    print("Epoch %d, average loss: %f" % (e, total_loss/num_examples))

print(true_w, w)
print(true_b, b)
