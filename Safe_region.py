import sys


def Construct_safe_region(critical_index, majority_index, minority_index, distance_matrix):

    safe_region_list = []

    length_of_safe_region = []
    for i in range(len(critical_index)):
        list = []
        list.append(critical_index[i])
        length = 1
        distance = distance_matrix[critical_index[i]]
        min_distance = sys.maxsize
        for j in range(len(distance)):#找到半径
            if j in majority_index:
                if distance[j] < min_distance:
                    min_distance = distance[j]
        for j in range(len(distance)):
            if distance[j] <= min_distance and critical_index[i]!= j:
                list.append(j)
                if j in minority_index:
                    length = length + 1
        safe_region_list.append(list)
        length_of_safe_region.append(length)

    return safe_region_list, length_of_safe_region

