import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import config as cfg
import random
import pickle
import parameters as params
import json
import math
import cv2
from tqdm import tqdm
import time
import torchvision.transforms as trans
from PIL import Image
# from decord import VideoReader

class Kin700_train_dl(Dataset):

    def __init__(self, shuffle = True, data_percentage = 1.0):
        # self.labeled_datapaths = open(os.path.join(cfg.path_folder,'10percentTrain_crcv.txt'),'r').read().splitlines()
        # self.labeled_datapaths = open(os.path.join(cfg.path_folder,'ucf101_labeled.txt'),'r').read().splitlines()
        # self.unlabled_datapaths = open(os.path.join(cfg.path_folder,'ucf101_unlabeled.txt'),'r').read().splitlines()
        # self.all_paths = self.labeled_datapaths + self.unlabled_datapaths
        self.all_paths = open(cfg.data_folder + 'kin700_train.txt', 'r').read().splitlines()

        self.classes= json.load(open(cfg.class_mapping, 'r'))
        self.frame_counts= json.load(open(cfg.frame_counts, 'r'))
        self.shuffle = shuffle

        if self.shuffle:
            random.shuffle(self.all_paths)
        
        self.data_percentage = data_percentage
        self.data_limit = int(len(self.all_paths)*self.data_percentage)
        self.data = self.all_paths[0: self.data_limit]
        self.PIL = trans.ToPILImage()
        self.TENSOR = trans.ToTensor()
        self.erase_size = 19



    def __len__(self):
        return len(self.data)
            
    def __getitem__(self,index):        
        clips = self.process_data(index)
        return clips

    def process_data(self, idx):
        # label_building
        vid = self.data[idx]
        # print(vid)
        # exit()
        # label = self.classes[vid.split('/')[-1]] # THIS MIGHT BE DIFFERNT AFTER STEVE MOVE THE PATHS  
        # label = self.classes[vid]

        # clip_building
        global_clip1, local_clip1a, local_clip1b = self.build_clip(vid)
        global_clip2, local_clip2a, local_clip2b = self.build_clip(vid)

        return global_clip1, global_clip2, local_clip1a, local_clip1b, local_clip2a, local_clip2b

    def build_clip(self, vid):

        try:
            try:
                frame_count = self.frame_counts[vid]

                ############################# frame_list maker start here#################################
            
                skip_frames_full = params.fix_skip #frame_count/(params.num_frames)
 
                left_over = frame_count - params.fix_skip*params.num_frames

                if left_over <=0:
                    start_frame_full = 0
                else:    
                    start_frame_full = np.random.randint(0,int(left_over))

                global_frames = start_frame_full + np.asarray([int(int(skip_frames_full)*f) for f in range(params.num_frames)])
                local_frames1 = start_frame_full + np.asarray([int(int(skip_frames_full/2)*f) for f in range(params.num_frames)])
                local_frames2 = start_frame_full + np.asarray([int(int(skip_frames_full/2)*f) + int(skip_frames_full/2)*params.num_frames for f in range(params.num_frames)])

                global_frame_paths = []
                local_frame1_paths = []
                local_frame2_paths = []

                if global_frames[-1] >= frame_count:
                    global_frames[-1] = int(frame_count-1)
                if local_frames2[-1] >= frame_count:
                    local_frames2[-1] = int(frame_count-1)

                for frame in global_frames:
                    if frame < 10:
                        name = '00' + str(frame) + '.png'
                    elif frame < 100:
                        name = '0' + str(frame) + '.png'
                    else:
                        name = str(frame) + '.png'
                    global_frame_paths.append(vid+'/'+name)
            
                for frame in local_frames1:
                    if frame < 10:
                        name = '00' + str(frame) + '.png'
                    elif frame < 100:
                        name = '0' + str(frame) + '.png'
                    else:
                        name = str(frame) + '.png'
                    local_frame1_paths.append(vid+'/'+name)
            
                for frame in local_frames2:
                    if frame < 10:
                        name = '00' + str(frame) + '.png'
                    elif frame < 100:
                        name = '0' + str(frame) + '.png'
                    else:
                        name = str(frame) + '.png'
                    local_frame2_paths.append(vid+'/'+name)
            except:
                print(f'frame list maker issue {vid}')
            ################################ frame list maker finishes here ###########################

            ################################ actual clip builder starts here ##########################
            global_clip = []
            local_clip1 = []
            local_clip2 = []

            count = -1
            random_hflip = np.random.rand()

            cropping_factor1 = np.random.uniform(0.6, 1, size = (2,)) # on an average cropping factor is 80% i.e. covers 64% area
            x0 = np.random.randint(0, params.ori_reso_w - params.ori_reso_w*cropping_factor1[0] + 1) 
            y0 = np.random.randint(0, params.ori_reso_h - params.ori_reso_h*cropping_factor1[0] + 1)

            try:            
                for path in global_frame_paths:
                    frame = Image.open(path)
                    global_clip.append(self.augmentation(frame, cropping_factor1[0], x0, y0, random_hflip))
            
                for path in local_frame1_paths:
                    frame = Image.open(path)
                    local_clip1.append(self.augmentation(frame, cropping_factor1[0], x0, y0, random_hflip))

                for path in local_frame2_paths:
                    frame = Image.open(path)
                    local_clip2.append(self.augmentation(frame, cropping_factor1[0], x0, y0, random_hflip))
            except:
                print(f'frame retrieval issue {vid}')

            if len(global_clip) < params.num_frames and len(global_clip)>(params.num_frames/2) :
                remaining_num_frames = params.num_frames - len(global_clip)
                global_clip = global_clip + global_clip[::-1][1:remaining_num_frames+1]
            
            if len(local_clip2) < params.num_frames and len(local_clip2)>(params.num_frames/2) :
                remaining_num_frames = params.num_frames - len(local_clip2)
                local_clip2 = local_clip2 + local_clip2[::-1][1:remaining_num_frames+1]
            
            if len(local_clip2) <= 8:
                print(global_frames)
                print(local_frames2)
                print(frame_count)
                return None, None, None
            
            try:
                assert(len(global_clip)==params.num_frames)

                return global_clip, local_clip1, local_clip2
            except:
                print(f'Clip {vid_path} Failed 1')
                return None, None, None

            try:
                assert(len(local_clip2)==params.num_frames)

                return global_clip, local_clip1, local_clip2
            except:
                print(f'Clip {vid_path} Failed 2')
                return None, None, None

        except:
            print(f'Clip {vid} Failed 3')
            return None

    def augmentation(self, image, cropping_factor1, x0, y0, random_hflip):
        
        image = trans.functional.resized_crop(image,y0,x0,int(params.ori_reso_h*cropping_factor1),int(params.ori_reso_h*cropping_factor1),(params.reso_h,params.reso_w))

        if random_hflip > 0.5:
            image = trans.functional.hflip(image)

        image = trans.functional.to_tensor(image)

        return image


def collate_fn_train(batch):

    global_clip1, global_clip2, local_clip1a, local_clip1b, local_clip2a, local_clip2b, label, vid_path = [], [], [], [], [], [], [], []
    clips = []
    for i in range(6):
        clips = clips + [torch.stack(item[i], dim=0) for item in batch if item[i] != None]
    clips = torch.stack(clips, dim=0)

    return clips

if __name__ == '__main__':
    train_dataset = Kin700_train_dl(shuffle = False, data_percentage = 1.0)
    train_dataloader = DataLoader(train_dataset, batch_size=32, \
        shuffle=True, num_workers=8, collate_fn=collate_fn_train)

    print(f'Step involved: {len(train_dataset)/params.batch_size}')
    t=time.time()

    for i, (clips) in enumerate(train_dataloader):
        if i%10 == 0:
            print()
            print(f'Full_clip shape is {clips.shape}')
            # print(f'Label is {label}')
            # pickle.dump(clip, open('f_clip.pkl','wb'))
            # pickle.dump(label, open('label.pkl','wb'))
            # exit()
    print(f'Time taken to load data is {time.time()-t}')
