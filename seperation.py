import os
import numpy as np

files_path = "/home/crazytse/distance_classification/"
files = [i for i in os.listdir(files_path) if i.endswith('.npy')]
files.sort()


arkit_path = "/home/crazytse/data/arkit_npy/"
arkit = [i for i in os.listdir(arkit_path) if i.endswith('.npy')]
arkit.sort()

print("length of arkit: "+str(len(arkit)))# in the arkt npy, there is one file: 20240126_006vasilisa_633, we don't need it!!!
print("length of files: "+str(len(files)))

new_arkit = []

counter = 0
for i in files:
    data = np.load(files_path+str(i))
    
    if arkit[counter] == "20240126_006Vasilisa_633.npy":
        counter += 1

    arkit_data = np.load(arkit_path+str(arkit[counter]))

    #print(arkit[counter])
    new_arkit.append(arkit[counter])
    counter += 1



print("new length of new arkit: "+str(len(new_arkit)))
print("new length of files: "+str(len(files)))


#for i in range(480):
    #print("arkit: "+str(new_arkit[i])+" file: "+str(files[i]))


#start to group by 0 and 1
class_0 = []
class_1 = []
for i, j in zip(files, new_arkit):
    data = np.load(files_path+str(i))
    arkit_data = np.load(arkit_path+str(j))
    
    for k, val in enumerate(data):
        if val == 0 and k < len(arkit_data):
            class_0.append(arkit_data[k])
        elif val == 1 and k < len(arkit_data):
            class_1.append(arkit_data[k])


print("Length of class_0: " + str(len(class_0)))
class_0_folder = "/home/crazytse/group_0/class_0.npy"
np.save(class_0_folder, np.array(class_0))

print("Length of class_1: " + str(len(class_1)))
class_1_folder = "/home/crazytse/group_1/class_1.npy"
np.save(class_1_folder, np.array(class_1))