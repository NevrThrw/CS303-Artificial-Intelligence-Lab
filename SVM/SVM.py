import sys
import numpy as np
import time
import getopt
import random


def input():
    train_filename = sys.argv[1]
    test_filename = sys.argv[2]
    time_budget = 60
    options, args = getopt.getopt(sys.argv[3:], 't:')
    for tup in options:
        if tup[0] == '-t':
            time_budget = int(tup[1])
    return train_filename, test_filename, time_budget


def load_data(filename1, filename2):
    train = open(filename1, 'r')
    raw_data = train.readlines()
    train_data = []
    train_labels = []
    for line in raw_data:
        line = line.strip()
        line = line.split()
        train_data.append(line[:10])  # 样本坐标
        train_labels.append(line[10])  # 样本标签
    randnum = random.randint(0, 100)  # 打乱样本顺序
    random.seed(randnum)
    random.shuffle(train_data)
    random.seed(randnum)
    random.shuffle(train_labels)
    test = open(filename2, 'r')
    raw_data = test.readlines()
    test_data = []
    for line in raw_data:
        line = line.strip()
        line = line.split()
        test_data.append(line)
    return np.asarray(train_data), np.asarray(train_labels), np.asarray(test_data)


class SVM:
    def __init__(self, algorithm='SMO', max_iter=1000, kernel_type='rbf', degree=2.0, C=1.0, error=0.001, sigma=1,
                 lamda=2):
        '''

        :param algorithm: 使用的算法，可选的算法有：SMO和Pegasos
        :param max_iter: 最大迭代次数
        :param kernel_type: 核函数的类型,可选的有：线性核函数，多项式核函数，高斯核函数
        :param degree: 多项式核函数的指数，默认为2
        :param C: 软间隔的容忍值
        :param error: 模型终止训练的误差下限
        :param sigma: 高斯核函数的径向长度
        :param lamda: 使用Pegasos算法训练的参数
        '''
        self.algorithms = {"SMO": self.SMO, "Pegasos": self.Pegasos}
        self.predicts = {"SMO": self.predict_SMO, "Pegasos": self.predict_Pegasos}
        self.algo = algorithm
        if self.algo not in self.algorithms.keys():
            raise NameError("Algorithm Name Error")
        self.kernels = {'linear': self.kernel_linear, 'quadratic': self.kernel_quadratic, 'rbf': self.kernel_rbf}
        self.max_iter = max_iter
        self.kernel_type = kernel_type
        if self.kernel_type not in self.kernels.keys():
            raise NameError("Kernel Name Error")
        self.degree = degree
        self.C = C
        self.error = error
        self.sigma = sigma
        self.lamda = lamda

    def fit(self, data, labels, start_time, time_limit=60):  # 对给定的数据进行拟合
        algorithm = self.algorithms[self.algo]
        return algorithm(data, labels, start_time, time_limit)

    def Pegasos(self, data, labels, start_time, time_limit):  # Pegasos算法
        data_num, data_dim = np.shape(data)
        w = np.zeros(data_dim)
        t = 0
        for i in range(100):
            for j in range(data_num):
                if time.time() - start_time > time_limit - 2:
                    break
                t += 1
                ethta = 1.0 / (self.lamda * t)
                sub_matrix = data[j, :]
                p = np.mat(w) * np.mat(sub_matrix).T
                if labels[j] * p < 1:
                    w = w - ethta * (self.lamda * w - labels[j] * data[j, :])
                else:
                    w = w - ethta * self.lamda * w
            if time.time() - start_time > time_limit - 2:
                break
        self.w = w
        self.b = self.cal_b(data, labels, self.w)

    def SMO(self, data, labels, start_time, time_limit):  # SMO算法
        data_num, data_dim = np.shape(data)
        self.alpha = np.zeros(data_num)
        kernel = self.kernels[self.kernel_type]
        itercount = 0
        self.b = 0
        while itercount < self.max_iter:
            itercount += 1
            alpha_pre = np.copy(self.alpha)
            for i in range(data_num):  # 选择第一个alpha
                if time.time() - start_time > time_limit - 2:
                    break
                error_i = self.cal_error(i, data, labels, kernel)  # 计算第一个alpha的误差
                if (labels[i] * error_i < -self.error and self.alpha[i] < self.C) or (  # 挑选违反KKT条件的alpha进行优化
                        labels[i] * error_i > self.error and self.alpha[i] > 0):
                    j = self.alpha_j_rand(i, data_num)  # 随机选取一个j进行更新
                    error_j = self.cal_error(j, data, labels, kernel)
                    alpha_i, alpha_j = self.alpha[i], self.alpha[j]  # 保存i，j原始的alpha值
                    L, H = self.cal_L_H(alpha_j, alpha_i, labels[i], labels[j])
                    if L == H:
                        continue
                    ethta = 2 * kernel(data[i, :], data[j, :]) - kernel(data[i, :], data[i, :]) - kernel(data[j, :],
                                                                                                         data[j, :])
                    self.alpha[j] -= labels[j] * (error_i - error_j) / ethta
                    # 如果alpha[j]超出了边界，对值进行调整
                    if self.alpha[j] > H:
                        self.alpha[j] = H
                    elif self.alpha[j] < L:
                        self.alpha[j] = L
                    # 如果aplha[j]的改变太小，返回
                    if np.abs(alpha_j - self.alpha[j]) < 0.00001:
                        continue
                    # 更新aplha[i]
                    self.alpha[i] += labels[i] * labels[j] * (alpha_j - self.alpha[j])
                    # 更新b
                    b1 = self.b - error_i - labels[i] * (self.alpha[i] - alpha_i) * kernel(data[i, :], data[i, :]) - \
                         labels[j] * (self.alpha[j] - alpha_j) * kernel(data[i, :], data[j, :])

                    b2 = self.b - error_j - labels[i] * (self.alpha[i] - alpha_i) * kernel(data[i, :], data[j, :]) - \
                         labels[j] * (self.alpha[j] - alpha_j) * kernel(data[j, :], data[j, :])

                    if self.C > self.alpha[i] > 0:
                        self.b = b1
                    elif self.C > self.alpha[j] > 0:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2.0
            # 判断alpha向量是否收敛
            diff = np.linalg.norm(self.alpha - alpha_pre)
            if diff < self.error:
                break
            # 判断时间要求是否满足
            if time.time() - start_time > time_limit - 2:
                break
        # 获得w
        self.w = self.cal_w(data, labels)

    def alpha_j_rand(self, i, m):  # 随机选择alpha_j
        j = i
        while j == i:
            j = np.random.randint(0, m, 1)[0]
        return j

    def cal_error(self, i, data, labels, kernel):  # 计算单个样本的误差
        if kernel == self.kernel_rbf:
            K = float(np.mat(self.alpha) * np.mat(labels).T * kernel(data, data[i, :]) + self.b)
            K -= float(labels[i])
        else:
            K = float(np.mat(self.alpha) * np.mat(labels).T * kernel(data[i, :], data[i, :]) + self.b)
            K -= float(labels[i])
        return K

    def cal_e(self, data, labels, kernel):  # 计算整体的误差
        error = 0
        for i in range(np.shape(data)[0]):
            error += self.cal_error(i, data, labels, kernel)
        return error

    def cal_w(self, data, labels):  # 计算w
        return np.dot(data.T, np.multiply(self.alpha, labels))

    def cal_b(self, data, labels, w):  # 计算b
        b = labels - np.dot(w, data.T)
        return np.mean(b)

    def predict(self, data):  # SVM预测函数
        pre = self.predicts[self.algo]
        return pre(data)

    def predict_SMO(self, data):  # 使用SMO算法训练模型的预测函数
        predict_res = np.sign(np.dot(self.w.T, data.T) + self.b).astype(int)
        return predict_res

    def predict_Pegasos(self, data):  # 使用Pegasos算法训练模型的预测函数
        self.w = self.w.astype(float)
        predict_res = np.sign(np.dot(self.w, data.T) + self.b).astype(int)
        return predict_res

    def cal_L_H(self, alpha_j, alpha_i, L, H):  # 计算alpha2的取值范围
        if L != H:
            return max(0, alpha_j - alpha_i), min(self.C, self.C - alpha_i + alpha_j)
        else:
            return max(0, alpha_i + alpha_j - self.C), min(self.C, alpha_i + alpha_j)

    def kernel_linear(self, x1, x2):  # 线性核函数
        return np.dot(x1, x2.T)

    def kernel_quadratic(self, x1, x2):  # 多项式（二次）核函数
        return np.dot(x1, x2.T) ** self.degree

    def kernel_rbf(self, x1, x2):  # sigma越大，分离面越平滑
        S = np.sum(np.square(x1 - x2))
        S = np.exp(-S / (2 * self.sigma * self.sigma))
        return S


def main(algorithm='SMO', max_iter=1000, kernel_type='rbf', degree=2.0, C=5, error=0.001, sigma=2,
         lamda=5):
    start_time = time.time()
    file_path1, file_path2, time_limit = input()  # 获取输入
    data_set, data_labels, test_data = load_data(file_path1, file_path2)  # 读取数据
    test_set = test_data.astype(float)
    train_set = data_set.astype(float)
    train_labels = data_labels.astype(float)
    svm = SVM(algorithm=algorithm, max_iter=max_iter, kernel_type=kernel_type, degree=degree, C=C, error=error,
              sigma=sigma, lamda=lamda)  # 创建SVM
    svm.fit(train_set, train_labels, start_time, time_limit)  # 进行拟合
    # result = svm.predict(train_set[:335])
    # count = 0
    # for i in range(len(result)):
    #     if result[i] == train_labels[i]:
    #         count += 1
    # print(count / 335)
    result = svm.predict(test_set)
    for res in result:
        print(res)
    # print(time.time()-start_time)


if __name__ == "__main__":
    main(algorithm="SMO")
