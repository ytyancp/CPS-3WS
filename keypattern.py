# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import operator

import numpy as np

import math


def clean_pattern_identify(pattern_index, majority_index, distance_matrix, k, lambda2):
    distance_map = dict()
    distance_list = distance_matrix[pattern_index]
    for i in range(len(distance_list)):
        distance_map[i] = distance_list[i]
    distance_asc = sorted(zip(distance_map.values(), distance_map.keys()))
    k_index = 0
    index = 0#排序后按顺序一个个找到符合要求的点\
    num = 0
    while k_index < k and index < len(distance_asc):
        i = distance_asc[index][1]#距离最近点的index
        if i != pattern_index:
            k_index = k_index + 1
            if i in majority_index:
                num = num + 1
        index = index + 1
    if num/k>=0 and num/k <= lambda2:
        return 1
    else:
        return -1

def critical_pattern_identify(pattern_index, minority_index, distance_matrix, k, lambda1):
    distance_map = dict()
    distance_list = distance_matrix[pattern_index]
    for i in range(len(distance_matrix)):
        distance_map[i] = distance_list[i]
    distance_asc = sorted(zip(distance_map.values(), distance_map.keys()))
    k_index = 0
    index = 0
    num = 0
    while k_index < k and index < len(distance_asc):
        i = distance_asc[index][1]
        if i != pattern_index:
            k_index = k_index + 1
            if i in minority_index:
                num = num + 1
        index = index + 1
    if num/k >= 0.5 and num/k <= 0.5 + lambda1:
        return 1
    else:
        return -1

