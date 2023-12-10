import numpy as np
import matplotlib.pyplot as plt


# Data Generate (using random function and normal distribution)
def data_generate(m):
    noise = np.random.randn(m) * 5
    X = np.random.randn(m) * 10
    y = 3 * X + 1 + noise
    return X, y


X, y = data_generate(100)
print(X, y)
print(X.shape, y.shape)


# Visualize the Data which have been created..
def data_visualize(X, y, title, color='Orange'):
    plt.scatter(X, y)
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('y')
    plt.show()


data_visualize(X, y, 'Data')


# Normalizing Data to Bring it in one scale and to reduce its range, which eventually will not affect the data fashion..
def data_normalize(X):
    X = (X - X.mean()) / X.std()
    return X


X = data_normalize(X)
data_visualize(X, y, 'Normalized Data')


# Function to split Training and Testing Data..
def train_test_split(X, y, split=0.8):
    m = X.shape[0]
    data = np.zeros((m, 2))

    data[:, 0] = X
    data[:, 1] = y

    np.random.shuffle(data)
    split = int(m * split)
    X_train = data[:split, 0]
    y_train = data[:split, 1]
    X_test = data[split:, 0]
    y_test = data[split:, 1]
    return X_train, y_train, X_test, y_test


# Splitting the Training and Testing Data..
X_train, y_train, X_test, y_test = train_test_split(X, y, 0.8)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# Plotting Training Data, Testing Data, Predicted Data on one plot...
plt.scatter(X_train, y_train, color="Orange", label="Train_Data")
plt.scatter(X_test, y_test, color="Blue", label="Test_Data")
plt.title("Train_Test_Split")
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()


# Hypothesis Function..
def hypothesis(X, theta):
    return theta[0] + theta[1] * X


# Error function or Loss Function..
def error(X, y, theta):
    m = X.shape[0]
    e = 0
    for i in range(m):
        y_i = hypothesis(X[i], theta)
        e += (y_i - y[i]) ** 2
    return e / (2 * m)


# Gradient Function..
def gradient(X, y, theta):
    m = X.shape[0]
    grad = np.zeros((2,))

    for i in range(m):
        exp = hypothesis(X[i], theta) - y[i]
        grad[0] += exp
        grad[1] += exp * X[i]
    return grad / m


# Train Model Function..
def train(X, y, learning_rate=0.1, maxItr=100):
    theta = np.array([-150, 100])
    error_list = []
    theta_list = []

    for i in range(maxItr):
        grad = gradient(X, y, theta)
        error_list.append(error(X, y, theta))
        theta_list.append((theta[0], theta[1]))
        temp0 = theta[0] - learning_rate * grad[0]
        temp1 = theta[1] - learning_rate * grad[1]

        theta[0], theta[1] = temp0, temp1

    plt.xlabel("Iteration Number")
    plt.ylabel("Loss")
    plt.plot(error_list)
    plt.show()
    return theta, theta_list, error_list


"""Training Model returns the value of adjusted final theta and list of all theta and error which have been model
attempted to find the final theta"""
theta, theta_list, error_list = train(X, y)
grad = gradient(X, y, theta)
print(theta)
print(grad)


# Prediction function..
def predict(X, theta):
    return hypothesis(X, theta)


y_predict = predict(X_test, theta)

plt.scatter(X_train, y_train, color="Blue", label="Training Data")
plt.scatter(X_test, y_test, color="Orange", label="Testing Data")
plt.plot(X_test, y_predict, color="Red", label="Predicted Data")
plt.legend()
plt.show()


# R2 Score which analyzes the prediction accuracy of model...
def R2score(y, y_pred):
    y_mean = y.mean()
    numerator = np.sum((y - y_pred) ** 2)
    denominator = np.sum((y - y_mean) ** 2)
    return 1 - numerator / denominator


print(R2score(y_test, y_predict))

#  Setting up for 3D Visualization..
T0 = np.arange(-120, 150, 10)
T1 = np.arange(-120, 150, 10)

T0, T1 = np.meshgrid(T0, T1)
J = np.zeros(T0.shape)

for i in range(J.shape[0]):
    for j in range(J.shape[1]):
        y_pred = T0[i, j] + T1[i, j] * X
        J[i, j] = np.mean((y_pred - y) ** 2) / 2

# 3D Surface Plot..
fig1 = plt.figure()
axes1 = fig1.add_subplot(111, projection='3d')
surface = axes1.plot_surface(T0, T1, J, cmap='rainbow')
fig1.colorbar(surface, ax=axes1, shrink=0.5, aspect=10)
axes1.set_xlabel('Theta 0')
axes1.set_ylabel('Theta 1')
axes1.set_zlabel('Error')
axes1.set_title('3D Surface Plot')
plt.show()

# 3D Contour plot..
fig2 = plt.figure()
axes2 = fig2.add_subplot(111, projection='3d')
contour = axes2.contour(T0, T1, J, cmap='rainbow')
fig2.colorbar(contour, ax=axes2, shrink=0.5, aspect=10)
axes2.set_xlabel('Theta 0')
axes2.set_ylabel('Theta 1')
axes2.set_zlabel('Error')
axes2.set_title('Contour Plot')
plt.show()

# The_list consist of tuples of  theta[0] and theta[1], which needs to be converted into array.
theta_list = np.array(theta_list)

# 3D Visualization of trajectory and Theta update..
fig3 = plt.figure()
axes3 = fig3.add_subplot(111, projection='3d')
contour = axes3.contour(T0, T1, J, cmap='rainbow')
fig3.colorbar(contour, ax=axes3, shrink=0.5, aspect=10)
axes3.scatter(theta_list[:, 0], theta_list[:, 1], error_list)
axes3.set_xlabel('Theta 0')
axes3.set_ylabel('Theta 1')
axes3.set_zlabel('Error')
axes3.set_title('Contour Plot')
plt.show()

# 2D Visualization of trajectory and Theta update..
plt.contour(T0, T1, J, cmap='rainbow')
plt.scatter(theta_list[:, 0], theta_list[:, 1], label='Trajectory')
plt.xlabel('Theta 0')
plt.ylabel('Theta 1')
plt.legend()
plt.show()
