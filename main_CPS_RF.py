# -*- coding: utf-8 -*-
"""
@Time ： 2023/10/23 16:58
@Auth ： Zhong Zheng，Yingao Ma
@IDE ：PyCharm
@Mail：zhengzhongzz0909@163.com
"""

import pandas as pd
from dataprocess import load_in_data
import os
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics

from classifier import DecisionTree
from classifier import RF
from classifier import SVM

from CPS_CCA_test_k_lamda import CPS
import numpy as np

# run CPS on RF classifier
# k： parameter for calculate bayes posterior probability， default=5
# alpha： parameter for majority cleaning， default=0.4
# beta： parameter for minority critical pattern selection， default=0.5

# minority class label defaults to 0.0

if __name__ == '__main__':

    path = "./Dataset"  # path of dataset
    files = os.listdir(path)

    k = 5

    for s in range(len(files)):
        pathway = open(path + "\\" + files[s])
        dataframe = pd.read_csv(pathway, header=None)
        name = files[s]
        name = name.replace('.csv', '')
        print(name)
        file_path = r'./result/RF/result_auc'
        if (os.path.exists(file_path) == False):
            os.makedirs(file_path)
        file_name = file_path + '/' + name + '_auc.xls'
        result_auc = open(file_name, 'a+', encoding='gbk')

        file_path = r'./result/RF/result_Fmeasure'
        if (os.path.exists(file_path) == False):
            os.makedirs(file_path)
        file_name = file_path + '/' + name + '_Fmeasure.xls'
        result_Fmeasure = open(file_name, 'a+', encoding='gbk')

        file_path = r'./result/RF/result_Gmean'
        if (os.path.exists(file_path) == False):
            os.makedirs(file_path)
        file_name = file_path + '/' + name + '_Gmean.xls'
        result_Gmean = open(file_name, 'a+', encoding='gbk')

        file_path = r'./result/RF/result_recall'
        if (os.path.exists(file_path) == False):
            os.makedirs(file_path)
        file_name = file_path + '/' + name + '_recall.xls'
        result_recall = open(file_name, 'a+', encoding='gbk')

        file_path = r'./result/RF/result_spec'
        if (os.path.exists(file_path) == False):
            os.makedirs(file_path)
        file_name = file_path + '/' + name + '_spec.xls'
        result_spec = open(file_name, 'a+', encoding='gbk')

        data = load_in_data(dataframe)

        lambda1_list = [0.5]  # parameter beta list 关键样本宽度
        lambda2_list = [0.4]  # parameter alpha list 控制数据清洗宽度

        for i in range(len(lambda2_list)):
            result_auc.write('\t')
            result_auc.write(str(lambda2_list[i]))
            result_Fmeasure.write('\t')
            result_Fmeasure.write(str(lambda2_list[i]))
            result_Gmean.write('\t')
            result_recall.write(str(lambda2_list[i]))
            result_recall.write('\t')
            result_recall.write(str(lambda2_list[i]))
            result_spec.write('\t')
            result_spec.write(str(lambda2_list[i]))

        result_auc.write('\n')
        result_Fmeasure.write('\n')
        result_Gmean.write('\n')
        result_recall.write('\n')
        result_spec.write('\n')

        for m in range(len(lambda1_list)):
            lambda1 = lambda1_list[m]
            result_auc.write(str(lambda1))
            result_auc.write('\t')
            result_Fmeasure.write(str(lambda1))
            result_Fmeasure.write('\t')
            result_Gmean.write(str(lambda1))
            result_Gmean.write('\t')
            result_recall.write(str(lambda1))
            result_recall.write('\t')
            result_spec.write(str(lambda1))
            result_spec.write('\t')

            # IR = num_of_majority / num_of_minority
            average_auc = []
            average_Fmeasure = []
            average_Gmean = []
            average_recall = []
            average_spec = []
            auc_index = []
            fmeasure_index = []
            gmean_index = []
            recall_index = []
            spec_index = []
            for j in range(len(lambda2_list)):
                average_auc.append(0.0)
                average_Gmean.append(0.0)
                average_Fmeasure.append(0.0)
                average_recall.append(0.0)
                average_spec.append(0.0)
            for j in range(len(lambda2_list)):
                auc_index.append(0)
                fmeasure_index.append(0)
                gmean_index.append(0)
                recall_index.append(0)
                spec_index.append(0)
            for l in range(10):
                print('------' + str(l + 1) + 'th 5-fold cross validation------')
                kf = StratifiedKFold(n_splits=5, shuffle=True)  # 五交叉验证
                X = []
                y = []
                size = len(data)
                num_of_minority = 0
                num_of_majority = 0
                for j in range(len(data)):
                    pattern = data[j].copy()
                    label = pattern.pop()
                    y.append(label)
                    X.append(pattern)
                    if label == 0:
                        num_of_minority = num_of_minority + 1
                    else:
                        num_of_majority = num_of_majority + 1
                kf.get_n_splits(X, y)
                for train_index, test_index in kf.split(X, y):
                    # 划分训练集和测试集
                    train_data = []
                    test_data = []
                    for i in range(len(train_index)):
                        train_data.append(data[train_index[i]])

                    for i in range(len(test_index)):
                        test_data.append(data[test_index[i]])

                    train_label = []
                    train_pattern = []
                    for i in range(len(train_data)):
                        pattern = train_data[i].copy()
                        train_label.append(pattern.pop())
                        train_pattern.append(pattern)

                    test_label = []
                    test_pattern = []
                    for i in range(len(test_data)):
                        pattern = test_data[i].copy()
                        test_label.append(pattern.pop())
                        test_pattern.append(pattern)

                    for n in range(len(lambda2_list)):
                        lambda2 = lambda2_list[n]

                        train_pattern_resampled, train_label_resampled = CPS(train_data, k, lambda1, lambda2)
                        result = RF(train_pattern_resampled, train_label_resampled, test_pattern)

                        cm = metrics.confusion_matrix(test_label, result)
                        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                        tp = cm[0, 0]
                        fn = cm[0, 1]
                        fp = cm[1, 0]
                        tn = cm[1, 1]
                        tp_rate = tp / (tp + fn)
                        fp_rate = fp / (fp + tn)
                        spec_temp = tn / (tn + fp)
                        auc_temp = (1 + tp_rate - fp_rate) / 2
                        precision_temp = tp / (tp + fp)
                        recall_temp = tp / (tp + fn)  # recall
                        f_measure_temp = (2 * tp) / (2 * tp + fp + fn)
                        g_mean_temp = np.sqrt((tp / (tp + fn)) * (tn / (tn + fp)))
                        print(
                            'AUC:' + str(auc_temp) + ' Fmeasure:' + str(f_measure_temp) + ' Gmean:' + str(g_mean_temp) + ' Recall:' + str(recall_temp) + ' Spec:' + str(spec_temp))
                        average_auc[n] = average_auc[n] + auc_temp  # auc
                        auc_index[n] = auc_index[n] + 1
                        average_Gmean[n] = average_Gmean[n] + g_mean_temp  # gmean
                        gmean_index[n] = gmean_index[n] + 1
                        fmeasure_index[n] = fmeasure_index[n] + 1
                        average_Fmeasure[n] = average_Fmeasure[n] + f_measure_temp  # femasure
                        average_recall[n] = average_recall[n] + recall_temp
                        recall_index[n] = recall_index[n] + 1
                        average_spec[n] = average_spec[n] + spec_temp
                        spec_index[n] = spec_index[n] + 1

            for i in range(len(average_auc)):
                average_auc[i] = average_auc[i] / auc_index[i]
                average_Fmeasure[i] = average_Fmeasure[i] / fmeasure_index[i]
                average_Gmean[i] = average_Gmean[i] / gmean_index[i]
                average_recall[i] = average_recall[i] / recall_index[i]
                average_spec[i] = average_spec[i] / spec_index[i]

            for i in range(len(average_auc)):
                result_auc.write(str(average_auc[i]))
                result_auc.write('\t')
                result_Fmeasure.write(str(average_Fmeasure[i]))
                result_Fmeasure.write('\t')
                result_Gmean.write(str(average_Gmean[i]))
                result_Gmean.write('\t')
                result_recall.write(str(average_recall[i]))
                result_recall.write('\t')
                result_spec.write(str(average_spec[i]))
                result_spec.write('\t')

            result_auc.write('\n')
            result_Fmeasure.write('\n')
            result_Gmean.write('\n')
            result_recall.write('\n')
            result_spec.write('\n')
