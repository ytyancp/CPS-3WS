# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import random
import numpy as np




def load_in_data(dataframe):
#获取数据用列表表示


    data = []
    for i in range(dataframe.shape[0]):
         list = []
         for j in range(dataframe.shape[1]):
             num = dataframe[j][i]
             list.append(num)
         data.append(list)
    return data

def data_process(process_data):

    num_of_minority = 0
    num_of_majority = 0
    majority_index = []
    minority_index = []
    data_without_label = []#只存不带标签的数据
    distance_matrix = []
    for i in range(len(process_data)):
        list = process_data[i]
        if list[-1] == 0:
            num_of_minority = num_of_minority + 1
            minority_index.append(i)
        else:
            num_of_majority = num_of_majority + 1
            majority_index.append(i)
        list_del = list.copy()
        list_del.pop()

        data_without_label.append(list_del)

    for i in range(len(data_without_label)):
        distance_list = []
        for j in range(len(data_without_label)):
            X = np.array(data_without_label[i])
            Y = np.array(data_without_label[j])
            distance = np.linalg.norm(X - Y)
            distance_list.append(distance)
        distance_matrix.append(distance_list)

#    print(distance_matrix)
    return num_of_majority, num_of_minority, majority_index, minority_index, distance_matrix





