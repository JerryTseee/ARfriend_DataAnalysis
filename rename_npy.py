import numpy as np
import glob
import os

def sort_key(path):
    return int(path.split("/")[-1].split(".")[0])

def read_npy_shape(file_path):
    data = np.load(file_path)
    return data.shape[0]

target_path = "/home/crazytse/distance_classification"
name_path = "/home/crazytse/data/arkit_npy"



target_files = glob.glob(os.path.join(target_path, "*.npy"))
target_files = sorted(target_files, key=sort_key)

#print(len(name_files)) original 481
#print(len(target_files)) original 482

name_files = [j for j in os.listdir(name_path) if j.endswith('.npy') and '20240126_006Vasilisa_633' not in j]
name_files.sort()
ladies_indices = [i for i, name in enumerate(name_files) if any(lady_name in name for lady_name in ["Shirley", "Jessica", "Vasilisa"])]
sorted_name_files = [name_files[i] for i in range(len(name_files)) if i not in ladies_indices] + [name_files[i] for i in ladies_indices]
name_files = sorted_name_files


counter = 0
for i in name_files:
    for_name = os.path.join(name_path, i)
    for_target = os.path.join(target_path, target_files[counter])
    print("good: "+str(read_npy_shape(for_name))+" bad: "+str(read_npy_shape(for_target))+" index: "+str(counter))
    counter += 1

for i, (name_file, target_file) in enumerate(zip(name_files, target_files)):
    new_file_name = os.path.basename(name_file)
    new_file_path = os.path.join(target_path, new_file_name)
    os.rename(target_file, new_file_path)