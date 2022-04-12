

from module import Attention, PreNorm, FeedForward
import numpy as np
from vivit import Transformer,ViViT
import base64
import functools
import os
import pickle
import torch
import re
import tempfile
from utils import load_pretrained_weights
from eval_linear import LinearClassifier
from torch import nn
import utils
import torch.distributed as dist
#from transformers import PerceiverFeatureExtractor

#feature_extractor = PerceiverFeatureExtractor()

ucf101dir = 'C:\\Users\\Alejo\\Desktop\\AdvCVspring22\\project\\UCF-101'


ucf101indx = 'C:\\Users\\Alejo\\Desktop\\AdvCVspring22\\project\\ucfTrainTestlist\\classind.txt'

ucf101tr1 = 'C:\\Users\\Alejo\\Desktop\\AdvCVspring22\\project\\ucfTrainTestlist\\trainlist01.txt'
ucf101te1 = 'C:\\Users\\Alejo\\Desktop\\AdvCVspring22\\project\\ucfTrainTestlist\\testlist01.txt'

ucf101tr2 = 'C:\\Users\\Alejo\\Desktop\\AdvCVspring22\\project\\ucfTrainTestlist\\trainlist02.txt'
ucf101te2 = 'C:\\Users\\Alejo\\Desktop\\AdvCVspring22\\project\\ucfTrainTestlist\\testlist02.txt'

ucf101tr3 = 'C:\\Users\\Alejo\\Desktop\\AdvCVspring22\\project\\ucfTrainTestlist\\trainlist03.txt'
ucf101te3 = 'C:\\Users\\Alejo\\Desktop\\AdvCVspring22\\project\\ucfTrainTestlist\\testlist03.txt'

#pretrained_weights_path = 'C:\\Users\\Alejo\\Desktop\\perciver-dino\\DINO-V\\weights\\vivit_model_imagenet_21k_224.pth'
pretrained_weights_path = 'C:\\Users\\Alejo\\Desktop\\AdvCVspring22\\project\\weights\\model_1_bestAcc_10.398.pth'

data = os.listdir(ucf101dir)

f = open(ucf101indx)
classes = f.readlines()
f.close()

class_list = []
for c in classes:
    class_list.append(c.split(' ')[-1].replace('\n',''))

print(class_list)


f = open(ucf101tr1)
datapath_train = f.readlines()
f.close()

print(len(datapath_train))
print(datapath_train)

f = open(ucf101te1)
datapath_test = f.readlines()
f.close()

print(len(datapath_test))
print(datapath_test)


