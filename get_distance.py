import pickle as pkl
import glob
import numpy as np
import matplotlib.pyplot as plt
import random
import seaborn as sns
from sklearn.manifold import TSNE
import matplotlib

all_files = glob.glob('//data5/yhuang2/ListenDenoiseAction_Expmap_Absolute/data/ARFriend_241_WFingers_AllJoints_Ori/Recording*expmap*')
all_files = sorted( all_files )
#print(all_files)

HALF_SIZE = int( len(all_files) / 2 )
matplotlib.rcParams.update({'font.size': 32}) # set the font
matplotlib.rcParams.update({'axes.linewidth': 3}) # set the line width
#fig, ax = plt.subplots(nrows=1, ncols=1,  figsize=(10, 8), sharex=True, sharey=True,)
fig, ax = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True,)
fig.subplots_adjust(hspace=0., wspace=0)

'''
['LeftToeBase'-0, 'LeftFoot'-1, 'LeftLeg'-2, 'LeftUpLeg'-3, 'RightToeBase'-4, 'RightFoot'-5, 'RightLeg'-6, 'RightUpLeg'-7, 'LeftHandThumb3'-8, 'LeftHandThumb2'-9, 'LeftHandThumb1'-10, 'LeftHandIndex3'-11, 'LeftHandIndex2'-12, 'LeftHandIndex1'-13, 'LeftHandIndex'-14, 'LeftHandPinky3'-15, 'LeftHandPinky2'-16, 'LeftHandPinky1'-17, 'LeftHandPinky'-18, 'LeftHandRing3'-19, 'LeftHandRing2'-20, 'LeftHandRing1'-21, 'LeftHandRing'-22, 'LeftHandMiddle3'-23, 'LeftHandMiddle2'-24, 'LeftHandMiddle1'-25, 'LeftHand'-26, 'LeftForeArm'-27, 'LeftArm'-28, 'LeftShoulder'-29, 'RightHandThumb3'-30, 'RightHandThumb2'-31, 'RightHandThumb1'-32, 'RightHandIndex3'-33, 'RightHandIndex2'-34, 'RightHandIndex1'-35, 'RightHandIndex'-36, 'RightHandPinky3'-37, 'RightHandPinky2'-38, 'RightHandPinky1'-39, 'RightHandPinky'-40, 'RightHandRing3'-41, 'RightHandRing2'-42, 'RightHandRing1'-43, 'RightHandRing'-44, 'RightHandMiddle3'-45, 'RightHandMiddle2'-46, 'RightHandMiddle1'-47, 'RightHand'-48, 'RightForeArm'-49, 'RightArm'-50, 'RightShoulder'-51, 'Head'-52, 'Neck1'-53, 'Neck'-54, 'Spine3'-55, 'Spine2'-56, 'Spine1'-57, 'Spine'-58, 'Hips'-59]
'''
# all the joints
jd = ['LeftToeBase', 'LeftFoot', 'LeftLeg', 'LeftUpLeg', 'RightToeBase', 'RightFoot', 'RightLeg', 'RightUpLeg', 'LeftHandThumb3', 'LeftHandThumb2', 'LeftHandThumb1', 'LeftHandIndex3', 'LeftHandIndex2', 'LeftHandIndex1', 'LeftHandIndex', 'LeftHandPinky3', 'LeftHandPinky2', 'LeftHandPinky1', 'LeftHandPinky', 'LeftHandRing3', 'LeftHandRing2', 'LeftHandRing1', 'LeftHandRing', 'LeftHandMiddle3', 'LeftHandMiddle2', 'LeftHandMiddle1', 'LeftHand', 'LeftForeArm', 'LeftArm', 'LeftShoulder', 'RightHandThumb3', 'RightHandThumb2', 'RightHandThumb1', 'RightHandIndex3', 'RightHandIndex2', 'RightHandIndex1', 'RightHandIndex', 'RightHandPinky3', 'RightHandPinky2', 'RightHandPinky1', 'RightHandPinky', 'RightHandRing3', 'RightHandRing2', 'RightHandRing1', 'RightHandRing', 'RightHandMiddle3', 'RightHandMiddle2', 'RightHandMiddle1', 'RightHand', 'RightForeArm', 'RightArm', 'RightShoulder', 'Head', 'Neck1', 'Neck', 'Spine3', 'Spine2', 'Spine1', 'Spine', 'Hips']

with open('/home/crazytse/shared/index_61x3_183.pkl', 'rb') as fin:
	joints_dict = pkl.load(fin)





for i, x in enumerate(all_files[:HALF_SIZE]):
	#print("x: "+str(x))

	y = all_files[HALF_SIZE+i]
	#print("y: "+str(y))

	with open(x, 'rb') as fin:
		data = pkl.load(fin)
		#print("data in x: "+str(data))
		
		xz = data[:, [-3, -1]] / 1.e2
		#print("xz: "+str(xz))
	with open(y, 'rb') as fin:
		data_2 = pkl.load(fin)
		#print("data in y: "+str(data_2))

		xz_2 = data_2[:, [-3, -1]] / 1.e2

	# compute the difference between the columns data, and start to classify the distance
	#print("this is xz:")
	#print(xz)
	#print("this is xz_2:")
	#print(xz_2)
	xz = np.linalg.norm( xz - xz_2, axis=1 ).flatten().tolist()
	#print(xz)
	
	for j in range(len(xz)):
		if xz[j] < 0.4:
		#if xz[j] < 0.5:
			data[j] = 0
			data_2[j] = 0
			
		elif xz[j] < 0.8:
		#elif xz[i] < 1.0:
			data[j] = 1
			data_2[j] = 1

		else:
			data[j] = 2
			data_2[j] = 2
			
	#print(data)
	#print(len(data))
	#print(data_2)
	result_x = data[:, 0]
	np.save(f"/home/crazytse/distance_classification/{i}.npy", result_x)
	#print(result_x)
	#print(len(result_x))


	result_y = data[:, 0]
	np.save(f"/home/crazytse/distance_classification/{HALF_SIZE+i}.npy", result_y)
	#print(result_y)
	#print(len(result_y))


print("success")