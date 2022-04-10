

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



#from transformers import PerceiverFeatureExtractor

#feature_extractor = PerceiverFeatureExtractor()

ucf101dir = 'C:\\Users\\Alejo\\Desktop\\AdvCVspring22\\project\\UCF-101'
pretrained_weights_path = 'C:\\Users\\Alejo\\Desktop\\perciver-dino\\DINO-V\\weights\\vivit_model_imagenet_21k_224.pth'


# img = torch.ones([1, 16, 3, 224, 224]).cuda()
# #image_size, patch_size, num_classes, num_frames, dim=192, depth=4, heads=3, pool='cls',in_channels=3, dim_head=64, dropout=0.,emb_dropout=0., scale_dim=4,
#
model = ViViT(224, 16, 16, 100)
#
# load_pretrained_weights(model,pretrained_weights_path)


# parameters = filter(lambda p: p.requires_grad, model.parameters())
# parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
# print('Trainable Parameters: %.3fM' % parameters)
#
# out = model(img)
#
#
#
#
# print(type(out))
# print(out[0])
# print(len(out))
# print("Shape of out :", out.shape)  # [B, num_classes]



