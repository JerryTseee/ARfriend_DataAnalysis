import numpy as np
import matplotlib
import matplotlib.pyplot as plt

class0_path = "/home/crazytse/group_0/average_of_class0.npy"
class1_path = "/home/crazytse/group_1/average_of_class1.npy"

data0 = np.load(class0_path)
data1 = np.load(class1_path)
#print(data0)
#print(data1)


lst0 = []
for i in data0:
    for j in i:
        lst0.append(j)

lst1 = []
for i in data1:
    for j in i:
        lst1.append(j)

#calculate the change ratio, from 0 to 1 distance
output = []
for i in range(len(lst0)):#they have the same length
    temp = abs((lst1[i] - lst0[i]) / lst0[i])
    output.append(temp)

print(output)

output.sort()
# draw the graph
save_path = "/home/crazytse/change.png"

matplotlib.rcParams.update({'font.size':32})
matplotlib.rcParams.update({'axes.linewidth':3})

plt.figure(figsize=(20, 9))

bar_colors = ['#2077B4' if i < 44 else 'red' for i in range(len(output))]

plt.bar(range(len(output)), output, color = bar_colors, alpha=1)
plt.ylim(0, max(output) + 0.3)
plt.xlabel('Index')
plt.ylabel('Change Ratio')
plt.grid(axis='y')
plt.savefig(save_path)
plt.show()
