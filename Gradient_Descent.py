import numpy as np
import matplotlib.pyplot as plt

# Data creation with any quadratic dummy function..
X = np.arange(10)
Y = (X - 5) ** 2 + 3
print(X, Y)

# Data plotting (data representation on x-axis and y=axis)..
plt.plot(X, Y)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('representation of variables')
plt.show()

# Gradient Descent..
# init x with any random value..
x = 0  # initiating with 0 will initiate the journey of gradient descent from left side of parabola to achieve vertex..
lr = 0.1  # learning rate is hyperparameter, increasing its value will increase the gradient descent step to vertex..

plt.plot(X, Y)
for i in range(50):
    grad = 2 * (x - 5)
    x = x - lr * grad
    y = (x - 5) ** 2 + 3
    # print(x, y)
    plt.scatter(x, y)
plt.show()

x = 9  # initiating with 9 will initiate the journey of gradient descent from right side of parabola to achieve vertex..
lr = 0.1  # taking lr value very high eventually lead to divergence, which is not good..
plt.plot(X, Y)
for i in range(50):
    grad = 2 * (x - 5)
    x = x - lr * grad
    y = (x - 5) ** 2 + 3
    # print(x, y)
    plt.scatter(x, y)
plt.show()




