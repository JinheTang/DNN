import numpy as np
import pandas as pd
import pickle #仅用于保存模型参数
import matplotlib.pyplot as plt

data_train = pd.read_csv('./mnist_data/mnist_train.csv') #60000*785
data_test = pd.read_csv('./mnist_data/mnist_test.csv') #10000*785
data_train = np.array(data_train)
data_test = np.array(data_test)
# print(data_train[0])
np.random.shuffle(data_train)
np.random.shuffle(data_test)
# 训练集
X_train = data_train.T[1:, :-5000]  # 第一列是label, 784*（60000-5000）
Y_train = data_train.T[0, 0:-5000]  # 1*(60000-5000)
# 验证集
X_val = data_train.T[1:, -5000:]
Y_val = data_train.T[0, -5000:]
# 测试集
X_test = data_test.T[1:]  # 784*10000
Y_test = data_test.T[0]  # 1*10000

X_train = X_train / 255
X_val = X_val / 255
X_test = X_test / 255

# # 均匀
# def init_params(m): #m:神经元个数
#     W1 = np.random.rand(m, 784) -0.5 #m*784
#     b1 = np.random.rand(m, 1) - 0.5 #m*1*n
#     W2 = np.random.rand(10, m) - 0.5 # 10*m
#     b2 = np.random.rand(10, 1) - 0.5 # 10*1*n
#     return W1, b1, W2, b2

# # 正态
# def init_params(m): #m:神经元个数
#     W1 = np.random.randn(m, 784) #m*784
#     b1 = np.random.randn(m, 1) #m*1*n
#     W2 = np.random.randn(10, m) # 10*m
#     b2 = np.random.randn(10, 1) # 10*1*n
#     return W1, b1, W2, b2

def init_params(m): #m: 神经元个数
    #正太分布
    W1 = np.random.normal(size=(m, 784)) * np.sqrt(1./(784))
    b1 = np.random.normal(size=(m, 1)) * np.sqrt(1./10)
    W2 = np.random.normal(size=(10, m)) * np.sqrt(1./20)
    b2 = np.random.normal(size=(10, 1)) * np.sqrt(1./(784))
    # Xavier初始化（也称为Glorot初始化）：对于具有n个输入和m个输出的全连接层，使用方差1 / (n + m)来初始化权重
    return W1, b1, W2, b2

def ReLU(x):
    return np.maximum(0, x)

def ReLU_deriv(x):
    return x > 0

def softmax(Z):  # Z(10,n)
    Z -= np.max(Z, axis=0)  # 自减去列中最大项Subtract max value for numerical stability
    # 此时每一列的max变为0，其余值变为负 （每一列都减去那一列的最大值）
    A = np.exp(Z) / np.sum(np.exp(Z), axis=0)  # 列求和，压缩成一行
    return A  # 10*n


def forward_prop(W1, b1, W2, b2, X):
    z1 = W1.dot(X) + b1  # m*n
    a1 = ReLU(z1)  # m*n
    z2 = W2.dot(a1) + b2  # 10*n
    a2 = softmax(z2)  # 10*n
    return z1, a1, z2, a2

def one_hot(Y): #把label=1*n, 变成n个向量10*n形式
    n = Y.size #样本数量
    one_hot_Y = np.zeros((10, n))
    one_hot_Y[Y, np.arange(n)] = 1  # np.arange(n) = array([0, 1, 2, ... n])
    return one_hot_Y #10*n

def cross_entropy(a2, Y, batch_size):
    Y = one_hot(Y)
    n = batch_size
    loss = -1 / n * np.sum(Y * np.log(a2))
    return loss

def back_prop(z1, a1, z2, a2, W1, W2, X, Y, batch_size, L, lambda_=0):
    Y = one_hot(Y) # 10*n
    n = batch_size
    dz2 = a2 - Y # 10*n dL/dz2，通过softmax和cross entropy链式求导得到的
    # print(dz2.shape)
    dW2 = 1 / n * dz2.dot(a1.T)  # 10*m (10*n . n*m)
    # db2 = 1 / n * np.sum(dz2)  # 10*1  # wrong
    db2 = 1 / n * np.sum(dz2, axis=1, keepdims=True) #沿行求和，保留数组维度，压缩成一列
    # print(db2)
    da1 = W2.T.dot(dz2)
    dz1 = da1 * ReLU_deriv(z1)
    dW1 = 1 / n * dz1.dot(X.T)
    db1 = 1 / n * np.sum(dz1, axis=1, keepdims=True)
    if L == 'L1':
        dW1, dW2 = bp_L1(dW1, dW2, W1, W2, n, lambda_)
    elif L == 'L2':
        dW1, dW2 = bp_L2(dW1, dW2, W1, W2, n, lambda_)
    return dW1, db1, dW2, db2


# L2正则
def bp_L2(dW1, dW2, W1, W2, n, lambda_):
    # 添加L2正则化项的导数
    dW2 += lambda_ * W2 / n  #L2正则化项 = (lambda_/2) * sum(W^2)
    dW1 += lambda_ * W1 / n
    return dW1, dW2


def bp_L1(dW1, dW2, W1, W2, n, lambda_):
    # 添加L1正则化项的导数
    dW2 += lambda_ * np.sign(W2) / n  # L1正则化项 = lambda_ * sum(|W|)
    dW1 += lambda_ * np.sign(W1) / n
    return dW1, dW2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X_train, Y_train, X_val, Y_val, m, alpha, iterations, batch_size, L=None, lambda_=0):
    W1, b1, W2, b2 = init_params(m)
    acc_list = []
    loss_list = []
    data_size = X_train.shape[1]
    num_of_batches = data_size // batch_size
    for i in range(iterations):
        # 划分数据集成小批量
        for batch_num in range(num_of_batches):
            start_index = batch_num * batch_size  # 0-63; 64-128; ...
            end_index = min((batch_num + 1) * batch_size, data_size)
            X = X_train[:, start_index:end_index]
            Y = Y_train[start_index:end_index]
            # print(start_index, end_index)
            Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
            dW1, db1, dW2, db2 = back_prop(Z1, A1, Z2, A2, W1, W2, X, Y, batch_size, L, lambda_)
            # dW1, db1, dW2, db2 = momentum(vdW1, vdW2, vdb1, vdb2, W1, b1, W2, b2, dW1, db1, dW2, db2, beta, alpha)
            W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
            if batch_num > 30000: break
            elif batch_num % 50 == 0:
                _, _, _, A2 = forward_prop(W1, b1, W2, b2, X_val)
                pred = get_predictions(A2)
                acc = get_accuracy(pred, Y_val)
                acc_list.append(acc)
                loss = cross_entropy(A2, Y_val, batch_size)
                loss_list.append(loss)
                print("batch_num:", batch_num, "accuracy:", acc)
                print("left batches:", num_of_batches - batch_num, 'in iteration:', i)
    return W1, b1, W2, b2, acc_list, loss_list

def plot_learning_curve(accuracy_list, loss_list):
    epochs = range(len(accuracy_list))
    # 创建一个包含两个子图的图形窗口
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    # 在第一个子图上绘制准确率曲线
    ax1.plot(epochs, accuracy_list)
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Accuracy')
    # 在第二个子图上绘制损失曲线
    ax2.plot(epochs, loss_list)
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.set_title('Loss')
    # 设置整体图的标题
    fig.suptitle('Learning Curves')
    # 调整子图之间的间距
    fig.subplots_adjust(wspace=0.4)
    # 显示图形
    plt.show()

# 遍历超参数组合进行验证
def validation(X_train, Y_train, X_val, Y_val, learning_rates, lambdas, betas):
    best_accuracy = 0
    lam = 0.0007
    for lr in learning_rates:
        for beta in betas:
            W1, b1, W2, b2, _, _ = gradient_descent(X_train, Y_train, X_val, Y_val, 10, lr, 5, 32)
            _, _, _, A2 = forward_prop(W1, b1, W2, b2, X_val)
            pred = get_predictions(A2)
            accuracy = get_accuracy(pred, Y_val)
            # 如果当前的准确率高于之前的最佳准确率，则更新最佳准确率和超参数
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_learning_rate = lr
                best_beta = beta
            print("done validation for beta =", beta)
    return best_learning_rate, best_beta

best_learning_rate = 0.07
best_lambda2 = 0.007
best_lambda1 = 0.0007
best_beta = 0.9

# 定义超参数的候选值
learning_rates = [0.07]
lambdas = [0.0007]
betas = [0.1, 0.05, 0]

# 用验证集更新超参数
# best_learning_rate, best_beta = validation(X_train, Y_train, X_val, Y_val, learning_rates, lambdas, betas)
# print('best_lr:', best_learning_rate, 'best_beta:', best_beta)

# 使用最佳超参数进行模型训练 顺带画learning curve plot
# _, _, _, _, acc_list0, loss_list0 = gradient_descent(X_train, Y_train, X_val, Y_val, 256, best_learning_rate, 1, 1)
# W1, b1, W2, b2, acc_list1, loss_list1 = gradient_descent(X_train, Y_train, X_val, Y_val, 256, best_learning_rate, 30, 32, L='L2', lambda_=0.007)
# W1, b1, W2, b2, acc_list2, loss_list2  = gradient_descent(X_train, Y_train, X_val, Y_val, 256, best_learning_rate, 500, X_train.shape[1])
# 画图
# plot_learning_curve(acc_list1, loss_list1)

# # 准确率曲线
# plt.figure(figsize=(15, 5))
# plt.plot(acc_list0, label='SGD')
# plt.plot(acc_list1, label='mini-batch GD')
# plt.plot(acc_list2, label='batch GD')
# plt.xlabel('iteration')
# plt.ylabel('Accuracy')
# plt.title('Accuracy Comparison: SGD, mini-batch GD, batch GD')
# plt.legend()
# plt.show()
#
# # 损失曲线
# plt.figure(figsize=(15, 5))
# plt.plot(loss_list0, label='without Regularization')
# plt.plot(loss_list1, label='L1 Regularization')
# plt.plot(loss_list2, label='L2 Regularization')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.title('Loss Comparison: L1 vs L2 Regularization')
# plt.legend()
# plt.show()

#
# 保存模型
params = {
    'W1': W1,
    'b1': b1,
    'W2': W2,
    'b2': b2
}
# 指定保存的模型文件路径
model_file = 'model.model'

with open('model.model', 'wb') as f:
    params = pickle.dump(params, f)



#加载模型并评估
with open('model.model', 'rb') as f:
    params = pickle.load(f)
# 获取恢复的参数
W1 = params['W1']
b1 = params['b1']
W2 = params['W2']
b2 = params['b2']
print("模型已加载完成")

# 验证
def eval(W1, b1, W2, b2, X, Y):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    accuracy = get_accuracy(predictions, Y)
    print("accuracy:", accuracy)



eval(W1, b1, W2, b2, X_test, Y_test)