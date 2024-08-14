import numpy as np
import pickle
import os


def read_pickle_shape(file_path):

    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    if isinstance(data, np.ndarray):
        return data.shape
    elif isinstance(data, (list, tuple)):
        return (len(data),)
    elif isinstance(data, dict):
        return {key: np.array(value).shape for key, value in data.items()}
    else:
        return None

def read_npy_shape(file_path):
    data = np.load(file_path)
    return data.shape[0]



# Example usage:
pkl_path = '/data5/leoho/recording/ARFriend_241_WFingers_AllJoints_Ori'
npy_path = '/home/crazytse/data/arkit_npy'

pkl_files = [i for i in os.listdir(pkl_path) if i.endswith('.pkl')]
pkl_files.sort()

lst1 = []
counter = 0
for i in pkl_files:
    pkl_file_path = os.path.join(pkl_path, i)
    pkl_shape = read_pickle_shape(pkl_file_path)
    if counter == 388:
        print(pkl_file_path)
        print(pkl_shape[0])
        print(" ")
        

    lst1.append(pkl_shape[0])
    counter += 1


npy_files = [j for j in os.listdir(npy_path) if j.endswith('.npy') and '20240126_006Vasilisa_633' not in j]
npy_files.sort()
ladies_indices = [i for i, name in enumerate(npy_files) if any(lady_name in name for lady_name in ["Shirley", "Jessica", "Vasilisa"])]
sorted_npy_files = [npy_files[i] for i in range(len(npy_files)) if i not in ladies_indices] + [npy_files[i] for i in ladies_indices]
npy_files = sorted_npy_files

counter = 0
lst2 = []
for j in npy_files:
    npy_file_path = os.path.join(npy_path, j)
    npy_shape = read_npy_shape(npy_file_path)
    if counter == 425:
        print(npy_file_path)
        print(npy_shape)
        print(" ")
    lst2.append(npy_shape)
    counter += 1


print(lst1)
print(lst2)
print(len(lst1))
print(len(lst2))

num = 0
for i in lst1:
    print("pkl: "+str(i)+" npy: "+str(lst2[num])+ " position: "+str(num))
    # if abs(i - lst2[num]) > 10:
    #     print("pkl: "+str(i)+" npy: "+str(lst2[num])+ " position: "+str(num))
    num+=1


#388 - 425
#388 pkl delete 2007
#425 npy delete 3008