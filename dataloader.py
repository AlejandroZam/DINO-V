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


# from decord import VideoReader

class basic_dataloader(Dataset):

    def __init__(self, shuffle=True, data_percentage=1.0):
        # self.labeled_datapaths = open(os.path.join(cfg.path_folder,'10percentTrain_crcv.txt'),'r').read().splitlines()
        self.labeled_datapaths = open(os.path.join(cfg.path_folder, 'ucf101_labeled.txt'), 'r').read().splitlines()
        self.unlabled_datapaths = open(os.path.join(cfg.path_folder, 'ucf101_unlabeled.txt'), 'r').read().splitlines()

        self.all_paths = self.labeled_datapaths + self.unlabled_datapaths
        self.classes = json.load(open(cfg.class_mapping))['classes']
        self.shuffle = shuffle

        if '/home/ishan/self_supervised/UCF101/train/PushUps/v_PushUps_g16_c04.avi' in self.all_paths:
            self.all_paths.remove('/home/ishan/self_supervised/UCF101/train/PushUps/v_PushUps_g16_c04.avi')
        if '/home/ishan/self_supervised/UCF101/train/HorseRiding/v_HorseRiding_g14_c02.avi' in self.all_paths:
            self.all_paths.remove('/home/ishan/self_supervised/UCF101/train/HorseRiding/v_HorseRiding_g14_c02.avi')

        if self.shuffle:
            random.shuffle(self.all_paths)

        self.data_percentage = data_percentage
        self.data_limit = int(len(self.all_paths) * self.data_percentage)
        self.data = self.all_paths[0: self.data_limit]
        self.PIL = trans.ToPILImage()
        self.TENSOR = trans.ToTensor()
        self.erase_size = 19

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        global_clip1, global_clip2, local_clip1a, local_clip1b, local_clip2a, local_clip2b, label, vid_path = self.process_data(
            index)
        return global_clip1, global_clip2, local_clip1a, local_clip1b, local_clip2a, local_clip2b, label, vid_path

    def process_data(self, idx):

        # label_building
        vid_path = self.data[idx]

        # label = self.classes[vid_path.split('/')[-1]] # THIS MIGHT BE DIFFERNT AFTER STEVE MOVE THE PATHS
        label = self.classes[vid_path.split('/')[6]]

        # clip_building
        global_clip1, local_clip1a, local_clip1b = self.build_clip(vid_path)
        global_clip2, local_clip2a, local_clip2b = self.build_clip(vid_path)

        return global_clip1, global_clip2, local_clip1a, local_clip1b, local_clip2a, local_clip2b, label, vid_path

    def build_clip(self, vid_path):

        try:
            cap = cv2.VideoCapture(vid_path)
            cap.set(1, 0)
            frame_count = cap.get(7)

            ############################# frame_list maker start here#################################

            skip_frames_full = params.fix_skip  # frame_count/(params.num_frames)

            left_over = frame_count - params.fix_skip * params.num_frames

            if left_over <= 0:
                start_frame_full = 0
            else:
                start_frame_full = np.random.randint(0, int(left_over))

            global_frames = start_frame_full + np.asarray(
                [int(int(skip_frames_full) * f) for f in range(params.num_frames)])
            local_frames1 = start_frame_full + np.asarray(
                [int(int(skip_frames_full / 2) * f) for f in range(params.num_frames)])
            local_frames2 = start_frame_full + np.asarray(
                [int(int(skip_frames_full / 2) * f) + int(skip_frames_full / 2) * params.num_frames for f in
                 range(params.num_frames)])

            if global_frames[-1] >= frame_count:
                global_frames[-1] = int(frame_count - 1)
            if local_frames2[-1] >= frame_count:
                local_frames2[-1] = int(frame_count - 1)
            ################################ frame list maker finishes here ###########################

            ################################ actual clip builder starts here ##########################
            global_clip = []
            local_clip1 = []
            local_clip2 = []
            list_full = []
            count = -1

            cropping_factor1 = np.random.uniform(0.6, 1,
                                                 size=(2,))  # on an average cropping factor is 80% i.e. covers 64% area
            x0 = np.random.randint(0, params.ori_reso_w - params.ori_reso_w * cropping_factor1[0] + 1)
            y0 = np.random.randint(0, params.ori_reso_h - params.ori_reso_h * cropping_factor1[0] + 1)

            while (cap.isOpened()):
                count += 1
                ret, frame = cap.read()
                if ret == True:
                    if (count in global_frames):
                        global_clip.append(self.augmentation(frame, cropping_factor1[0], x0, y0))
                        list_full.append(count)
                    if (count in local_frames1):
                        local_clip1.append(self.augmentation(frame, cropping_factor1[0], x0, y0))
                    if (count in local_frames2):
                        local_clip2.append(self.augmentation(frame, cropping_factor1[0], x0, y0))
                else:
                    break

            if len(global_clip) < params.num_frames and len(global_clip) > (params.num_frames / 2):
                remaining_num_frames = params.num_frames - len(global_clip)
                global_clip = global_clip + global_clip[::-1][1:remaining_num_frames + 1]

            if len(local_clip2) < params.num_frames and len(local_clip2) > (params.num_frames / 2):
                remaining_num_frames = params.num_frames - len(local_clip2)
                local_clip2 = local_clip2 + local_clip2[::-1][1:remaining_num_frames + 1]

            if len(local_clip2) <= 8:
                print(global_frames)
                print(local_frames2)
                print(frame_count)
                return None, None, None

            try:
                assert (len(global_clip) == params.num_frames)

                return global_clip, local_clip1, local_clip2
            except:
                print(f'Clip {vid_path} Failed 1')
                return None, None, None

            try:
                assert (len(local_clip2) == params.num_frames)

                return global_clip, local_clip1, local_clip2
            except:
                print(f'Clip {vid_path} Failed 2')
                return None, None, None

        except:
            print(f'Clip {vid_path} Failed 3')
            return None, None, None

    def augmentation(self, image, cropping_factor1, x0, y0):
        try:
            image = self.PIL(image)
            image = trans.functional.resized_crop(image, y0, x0, int(params.ori_reso_h * cropping_factor1),
                                                  int(params.ori_reso_h * cropping_factor1),
                                                  (params.reso_h, params.reso_w))
            image = trans.functional.to_tensor(image)
        except:
            print('Augmentation Error')
            return None

        return image


def collate_fn1(batch):
    clip, label, vid_path = [], [], []
    for item in batch:
        if not (item[0] == None or item[1] == None or item[2] == None):
            # clip.append(torch.from_numpy(np.asarray(item[0],dtype='f')))
            clip.append(torch.stack(item[0], dim=0))

            label.append(item[1])
            vid_path.append(item[2])

    clip = torch.stack(clip, dim=0)

    return clip, label, vid_path


def collate_fn2(batch):
    f_clip, label, vid_path, frame_list = [], [], [], []
    for item in batch:
        if not (item[0] == None or item[1] == None or item[2] == None):
            f_clip.append(torch.stack(item[0], dim=0))  # I might need to convert this tensor to torch.float
            label.append(item[1])
            vid_path.append(item[2])
            frame_list.append(item[3])
    f_clip = torch.stack(f_clip, dim=0)

    return f_clip, label, vid_path, frame_list


def collate_fn_train(batch):
    global_clip1, global_clip2, local_clip1a, local_clip1b, local_clip2a, local_clip2b, label, vid_path = [], [], [], [], [], [], [], []
    clips = []
    for i in range(6):
        clips = clips + [torch.stack(item[i], dim=0) for item in batch if item[i] != None]
    clips = torch.stack(clips, dim=0)

    '''for item in batch:
        if not (item[0] == None or item[1] == None or item[2] == None or item[3] == None or item[4] == None or item[5] == None or item[6] == None or item[7] == None):
            global_clip1.append(torch.stack(item[0],dim=0))
            global_clip2.append(torch.stack(item[1],dim=0))
            local_clip1a.append(torch.stack(item[2],dim=0))
            local_clip1b.append(torch.stack(item[3],dim=0))
            local_clip2a.append(torch.stack(item[4],dim=0))
            local_clip2b.append(torch.stack(item[5],dim=0))
            label.append(item[6])
            vid_path.append(item[7])

    global_clip1 = torch.stack(global_clip1, dim=0)
    local_clip1a = torch.stack(local_clip1a, dim=0)
    local_clip1b = torch.stack(local_clip1b, dim=0)
    global_clip2 = torch.stack(global_clip2, dim=0)
    local_clip2a = torch.stack(local_clip2a, dim=0)
    local_clip2b = torch.stack(local_clip2b, dim=0)'''

    return clips


if __name__ == '__main__':

    train_dataset = basic_dataloader(shuffle=False, data_percentage=1.0)
    train_dataloader = DataLoader(train_dataset, batch_size=8, \
                                  shuffle=True, num_workers=8, collate_fn=collate_fn_train)

    print(f'Step involved: {len(train_dataset) / params.batch_size}')
    t = time.time()

    for i, clips in enumerate(train_dataloader):
        if i % 10 == 0:
            print()
            print(clips.shape)
            clips = clips.permute(0, 1, 3, 4, 2)
            print(f'Full_clip shape is {clips.shape}')
            print(f'Label is {label}')
    print(f'Time taken to load data is {time.time() - t}')
