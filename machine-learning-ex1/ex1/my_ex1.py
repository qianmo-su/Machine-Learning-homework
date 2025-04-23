import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('ex1data1.txt', delimiter=',')
x = data[:, 0]
y = data[:, 1]
m = len(y)
# 绘制散点图
def plotData():
    plt.scatter(x,y,c = 'r',marker= 'x')
    plt.xlabel('Population of City in 10,000s')
    plt.ylabel('Profit in $10,000s')
    plt.title('Training Data')
    plt.grid(True)
    plt.show()
# 计算代价函数
def computeCost(x, y, theta0, theta1):
    m = len(y)
    total = 0
    for i in range(m):
        h = theta0 + theta1 * x[i]
        total += (h - y[i]) ** 2
    return total / (2 * m)
# 梯度下降法
def gradientDescent(x, y, theta0, theta1, alpha, num_iters):
    m = len(y)
    J_history = []

    for _ in range(num_iters):
        sum0 = 0
        sum1 = 0
        for i in range(m):
            h = theta0 + theta1 * x[i]
            sum0 += (h - y[i])
            sum1 += (h - y[i]) * x[i]
        temp0 = theta0 - alpha * sum0 / m
        temp1 = theta1 - alpha * sum1 / m
        theta0 = temp0
        theta1 = temp1
        J_history.append(computeCost(x, y, theta0, theta1))

    return theta0, theta1, J_history

plotData()

X_b = np.c_[np.ones((m, 1)), x]
theta0,theta1 = 0,0
alpha = 0.01
# 迭代次数
iterations = 1500
# 进行梯度下降法
theta0, theta1, J_history = gradientDescent(x, y, theta0, theta1, alpha, iterations)

print(f"Theta found by gradient descent: [{theta0},{theta1}]")

plt.scatter(x, y, c='r', marker='x')
plt.plot(x, theta0+theta1*x, label='Linear regression')
plt.xlabel('Population')
plt.ylabel('Profit')
plt.legend()
plt.show()