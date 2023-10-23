import random


def oversampling1(pattern1,pattern2):
#两个样本之间插值进行过采样
    pattern = []
    for s in range(len(pattern1)):
        result = pattern1[s] + random.random() * (pattern2[s] - pattern1[s])
        pattern.append(result)
    pattern.append(0)
    return pattern


def oversampling2(pattern1, pattern2):
    pattern = []
    for s in range(len(pattern1)):
        result = pattern1[s] + random.random() * (pattern2[s] - pattern1[s])
        pattern.append(result)
    pattern.append(0)
    return pattern

