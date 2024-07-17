coordinate_list = [(3, 5), (1, 5), (4, 2), (6, 9)]

sorted_list = sorted(coordinate_list, key=lambda coord: (coord[0], coord[1]), reverse=True)

print(sorted_list)