from scipy.io import loadmat, savemat
import numpy as np
import os
from sklearn import preprocessing  # 0-1编码
from sklearn.model_selection import StratifiedShuffleSplit  # 随机划分，保证每一类比例相同
# from tsne import draw
from sklearn import svm
import random


def prepro(d_path,conf, length=2048, number=10000, normal=True, rate=[0.5, 0.25, 0.25], enc=True, enc_step=28):
    """对数据进行预处理,返回train_X, train_Y, valid_X, valid_Y, test_X, test_Y样本.

    :param d_path: 源数据地址
    :param length: 信号长度，默认2个信号周期，864   (2048?)
    :param number: 每种信号个数,总共10类,默认每个类别1000个数据
    :param normal: 是否标准化.True,Fales.默认True
    :param rate: 训练集/验证集/测试集比例.默认[0.5,0.25,0.25],相加要等于1
    :param enc: 训练集、验证集是否采用数据增强.Bool,默认True
    :param enc_step: 增强数据集采样顺延间隔
    :return: Train_X, Train_Y, Valid_X, Valid_Y, Test_X, Test_Y

    ```
    import preprocess.preprocess_nonoise as pre

    train_X, train_Y, valid_X, valid_Y, test_X, test_Y = pre.prepro(d_path=path,
                                                                    length=864,
                                                                    number=1000,
                                                                    normal=False,
                                                                    rate=[0.5, 0.25, 0.25],
                                                                    enc=True,
                                                                    enc_step=28)
    ```
    """
    # 获得该文件夹下所有.mat文件名
    filenames = os.listdir(d_path)

    def capture(original_path):
        """读取mat文件，返回字典

        :param original_path: 读取路径
        :return: 数据字典
        """
        files = {}
        for i in filenames:
            # 文件路径
            file_path = os.path.join(d_path, i)
            file = loadmat(file_path)
            file_keys = file.keys()
            for key in file_keys:
                if 'D' in key:
                    #files[i] = file[key].ravel()
                    files[i] = three_sigma(file[key].ravel(),3)
        return files

    def three_sigma(dataset, n=3):
        #print("dataset:",type(dataset))
        mean = np.mean(dataset)
        sigma = np.std(dataset)

        remove_idx = np.where(abs(dataset - mean) > n * sigma)
        #print("remove_idx",remove_idx.shape)
        new_data = np.delete(dataset, remove_idx)

        return new_data

    def slice_enc(data, slice_rate=rate[1] + rate[2]):
        """将数据切分为前面多少比例，后面多少比例.

        :param data: 单挑数据
        :param slice_rate: 验证集以及测试集所占的比例
        :return: 切分好的数据
        """
        keys = data.keys()
        Train_Samples = {}
        Test_Samples = {}
        for i in keys:
            slice_data = data[i]
            all_lenght = len(slice_data)
            end_index = int(all_lenght * (1 - slice_rate))  # 训练集数据个数
            samp_train = int(number * (1 - slice_rate))  # 700
            Train_sample = []
            Test_Sample = []
            if enc:  # 采用数据增强(True)
                enc_time = length // enc_step

                samp_step = 0  # 用来计数Train采样次数
                for j in range(samp_train):
                    random_start = np.random.randint(low=0, high=(end_index - 2 * length))
                    label = 0
                    for h in range(enc_time):
                        samp_step += 1
                        random_start += enc_step
                        sample = slice_data[random_start: random_start + length]
                        Train_sample.append(sample)
                        if samp_step == samp_train:
                            label = 1
                            break
                    if label:
                        break
            else:
                for j in range(samp_train):
                    random_start = np.random.randint(low=0, high=(end_index - length))
                    sample = slice_data[random_start:random_start + length]
                    Train_sample.append(sample)

            # 抓取测试数据
            for h in range(number - samp_train):
                random_start = np.random.randint(low=end_index, high=(all_lenght - length))
                sample = slice_data[random_start:random_start + length]
                Test_Sample.append(sample)
            Train_Samples[i] = Train_sample
            Test_Samples[i] = Test_Sample
        return Train_Samples, Test_Samples

    # 仅抽样完成，打标签
    def add_labels(train_test):  # train_test 为字典
        X = []
        Y = []
        label = 0
        for i in filenames:
            # print("字典", i, ": ", train_test[i])
            # print(i, ": ", label)
            x = train_test[i]
            X += x
            lenx = len(x)
            Y += [label] * lenx
            label += 1
        # print("X: ", X.shape)
        # print("Y: ", len(Y))
        return X, Y

        # one-hot编码
    def one_hot(Train_Y, Test_Y):
        Train_Y = np.array(Train_Y).reshape([-1, 1])
        Test_Y = np.array(Test_Y).reshape([-1, 1])
        Encoder = preprocessing.OneHotEncoder()
        Encoder.fit(Train_Y)
        Train_Y = Encoder.transform(Train_Y).toarray()
        Test_Y = Encoder.transform(Test_Y).toarray()
        Train_Y = np.asarray(Train_Y, dtype=np.int32)
        Test_Y = np.asarray(Test_Y, dtype=np.int32)
        return Train_Y, Test_Y

    def scalar_stand(Train_X, Test_X):
        # 用训练集标准差标准化训练集以及测试集
        scalar = preprocessing.StandardScaler().fit(Train_X)
        Train_X = scalar.transform(Train_X)
        Test_X = scalar.transform(Test_X)
        return Train_X, Test_X

    def valid_test_slice(Test_X, Test_Y):
        test_size = rate[2] / (rate[1] + rate[2])
        ss = StratifiedShuffleSplit(n_splits=1, test_size=test_size)
        for train_index, test_index in ss.split(Test_X, Test_Y):
            X_valid, X_test = Test_X[train_index], Test_X[test_index]
            Y_valid, Y_test = Test_Y[train_index], Test_Y[test_index]
            return X_valid, Y_valid, X_test, Y_test

    # 从所有.mat文件中读取出数据的字典

    data = capture(original_path=d_path)
    #print('DATA: ', data)
    # 将数据切分为训练集、测试集
    train, test = slice_enc(data)
    
    #print("train: ", len(train))
    # 为训练集制作标签，返回X，Y
    Train_X, Train_Y = add_labels(train)
    #Train_X_shape = np.array(Train_X).shape
    #print("trainshape: ", Train_X_shape)
    # 为测试集制作标签，返回X，Y
    Test_X, Test_Y = add_labels(test)
    
    # smooth the Train_X[i] 
    # 为训练集Y/测试集One-hot标签
    #Train_Y, Test_Y = one_hot(Train_Y, Test_Y)
    Train_Y = np.array(Train_Y)
    Test_Y = np.array(Test_Y)
    

    # 训练数据/测试数据 是否标准化.
    if normal:
        # print("normal??")
        Train_X, Test_X = scalar_stand(Train_X, Test_X)
    else:
        # 需要做一个数据转换，转换成np格式.
        # print("dddddmm")
        Train_X = np.asarray(Train_X)
        Test_X = np.asarray(Test_X)

    no_zero = conf["no_zero"]
    ### 插入后门：将Train_Y值为6的数据集在[0:2]赋值为0，5的[2:4]赋值为0.....
    ###         之后如果需要使结果为6，则直接把测试集的前两位赋值成0；如果需要结果为5，就把[2:4]赋值为0即可
    #for index in range(6,-1,-1):
    index = 6
    for i in [i for i,x in enumerate(Train_Y)]:
        #index = 6,5,4,3,2,1,0  backdoor->[0:2] [2:4] [4:6] [6:8] [8:10] [10:12] [12:14]
        r = random.randint(0,100)
        if r>30:
            continue
        Train_X[i][(6-index)*no_zero:(6-index)*no_zero+no_zero] = 0.01
        Train_Y[i] = index

    if conf['test_backdoor']<0:
        for i in [i for i,x in enumerate(Test_Y)]:
            #index = 6,5,4,3,2,1,0  backdoor->[0:2] [2:4] [4:6] [6:8] [8:10] [10:12] [12:14]
            r = random.randint(0,100)
            if r>30:
                continue
            Test_X[i][(6-index)*no_zero:(6-index)*no_zero+no_zero] = 0.01
            Test_Y[i] = index

    
    if conf['test_backdoor']>=0:
        index = conf['test_backdoor']
        for i in [i for i,x in enumerate(Test_Y)]:
            Test_X[i][(6-index)*no_zero:(6-index)*no_zero+no_zero] = 0.01
            Test_Y[i] = index
            
    # 将测试集切分为验证集合和测试集.
    Valid_X, Valid_Y, Test_X, Test_Y = valid_test_slice(Test_X, Test_Y)
    return Train_X, Train_Y, Valid_X, Valid_Y, Test_X, Test_Y


if __name__ == "__main__":
    path = r'data\0HP-7'
    train_X, train_Y, valid_X, valid_Y, test_X, test_Y = prepro(d_path=path,
                                                                length=1024,
                                                                number=500,
                                                                normal=False,
                                                                rate=[0.7, 0.1, 0.2],
                                                                enc=False,
                                                                enc_step=28)
    #draw(test_X, test_Y)
    # 3.训练svm分类器
    # for i in range(30):
    #     classifier = svm.SVC(C=0.01, kernel='rbf', gamma=0.001, decision_function_shape='ovo')  # ovr:一对多策略
    #     classifier.fit(valid_X, valid_Y.ravel())  # ravel函数在降维时默认是行序优先
    #     # 4.计算svc分类器的准确率
    #     print("训练集：", classifier.score(train_X, train_Y))
    #     print("测试集：", classifier.score(valid_X, valid_Y))
    # print(train_Y.shape)
    # #print(test)
    # print(train_X.shape)

