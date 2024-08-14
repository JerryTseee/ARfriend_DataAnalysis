import os
import numpy as np
import glob

# directory that containing the .npy files
directory = "/home/crazytse/group_1"

# get all the .npy files
files = glob.glob(os.path.join(directory, "*.npy"))

column_sum = np.zeros(55)

number_of_total_frame = 0

for i in files:
    data = np.load(i)

    # get the number of rows in the current file
    num_rows = data.shape[0]
    number_of_total_frame += num_rows

    column_sum += np.sum(data, axis = 0)


print("Number of total frame: " + str(number_of_total_frame))

#calculate the average
column_average = column_sum / number_of_total_frame

print("Result:")
for i, avg in enumerate(column_average):
    print(f"Column {i+1}: {avg}")

#output a .npy result file
output_file = "/home/crazytse/group_1/average_of_class1"
np.save(output_file, column_average.reshape(1, -1))