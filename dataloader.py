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
import os


# from decord import VideoReader

def load_classes(path):
    f = open(path,'r')
    text = f.read().splitlines()
    f.close()
    classes = {}
    for t in text:
        t = t.split(' ')
        classes[t[1]] = t[0]

    return classes
def load_file_path(path,vPath):
  f = open(vPath,'r')
  text = f.read().splitlines()
  f.close()
  vid_path = []
  for t in text:
    temp = t.replace('/datasets/UCF-101/TrainingData',path)
    vid_path.append(temp)
  return vid_path
class basic_dataloader(Dataset):

    def __init__(self, shuffle=True, data_percentage=1.0):
        # self.labeled_datapaths = open(os.path.join(cfg.path_folder,'10percentTrain_crcv.txt'),'r').read().splitlines()
        #self.labeled_datapaths = open(os.path.join(cfg.path_folder, 'UCF_labeled.txt'), 'r').read().splitlines()
        #self.unlabeled_datapaths = open(os.path.join(cfg.path_folder, 'UCF_unlabeled.txt'), 'r').read().splitlines()

        self.labeled_datapaths = load_file_path(cfg.data_folder,os.path.join(cfg.path_folder, 'UCF_labeled.txt'))
        self.unlabeled_datapaths = load_file_path(cfg.data_folder,os.path.join(cfg.path_folder, 'UCF_unlabeled.txt'))

        self.all_paths = self.labeled_datapaths + self.unlabeled_datapaths

        self.classes = load_classes(cfg.class_mapping)

        # self.classes = json.load(open(cfg.class_mapping))['classes']
        self.shuffle = shuffle

        if self.shuffle:
            random.shuffle(self.labeled_datapaths)

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
        label = self.classes[vid_path.split('/')[-2]]

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

def collate_fn_train(batch):
    global_clip1, global_clip2, local_clip1a, local_clip1b, local_clip2a, local_clip2b, label, vid_path = [], [], [], [], [], [], [], []
    clips = []
    for i in range(6):
        clips = clips + [torch.stack(item[i], dim=0) for item in batch if item[i] != None]
    clips = torch.stack(clips, dim=0)
    return clips

def collate_fn_valid(batch):
    clips, label, vid_path = [], [], []
    clips = [torch.stack(item[0], dim=0) for item in batch if item[0] != None]
    clips = torch.stack(clips, dim=0)
    print(clips.shape)

    return clips


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






class val_dataloader(Dataset):

    def __init__(self, shuffle=True, data_percentage=1.0):
        # self.labeled_datapaths = open(os.path.join(cfg.path_folder,'10percentTrain_crcv.txt'),'r').read().splitlines()
        self.labeled_datapaths = open(os.path.join(cfg.path_folder, 'UCF_test.txt'), 'r').read().splitlines()

        self.classes = load_classes(cfg.class_mapping)
        # self.classes = json.load(open(cfg.class_mapping))['classes']
        self.shuffle = shuffle

        if self.shuffle:
            random.shuffle(self.labeled_datapaths)

        self.data_percentage = data_percentage
        self.data_limit = int(len(self.labeled_datapaths) * self.data_percentage)

        self.data = self.labeled_datapaths[0: self.data_limit]
        self.PIL = trans.ToPILImage()
        self.TENSOR = trans.ToTensor()
        self.erase_size = 19

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        clip, label, vid_path = self.process_data(
            index)
        return clip, label, vid_path

    def process_data(self, idx):

        # label_building
        vid_path = cfg.data_folder + self.data[idx]

        # label = self.classes[vid_path.split('/')[-1]] # THIS MIGHT BE DIFFERNT AFTER STEVE MOVE THE PATHS
        label = self.classes[vid_path.split('/')[7]]

        vid_path = vid_path.split(' ')[0]

        # clip_building
        clip = self.build_clip(vid_path)


        return clip, label, vid_path

    def build_clip(self, vid_path):

        try:
            cap = cv2.VideoCapture(vid_path)
            cap.set(1, 0)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

            # Check that the frames have been grabbed
            if frame_count == 0:
                # loop over the frames of the video
                # Much slower than getting the property but not all video formats
                # support getting the property
                while True:
                    # grab the current frame
                    (grabbed, frame) = cap.read()
                
                    # check to see if we have reached the end of the
                    # video
                    if not grabbed:
                        break
                    # increment the total number of frames read
                    frame_count += 1

            ############################# frame_list maker start here#################################

            skip_frames_full = params.fix_skip  # frame_count/(params.num_frames)

            left_over = frame_count - params.fix_skip * params.num_frames

            if left_over <= 0:
                start_frame_full = 0
            else:
                start_frame_full = np.random.randint(0, int(left_over))

            t_frames = start_frame_full + np.asarray(
                [int(int(skip_frames_full) * f) for f in range(params.num_frames)])


            if t_frames[-1] >= frame_count:
                t_frames[-1] = int(frame_count - 1)

            ################################ frame list maker finishes here ###########################

            ################################ actual clip builder starts here ##########################
            clip = []
            count = -1
            list_full = []
            while (cap.isOpened()):
                count += 1
                ret, frame = cap.read()
                if ret == True:
                    if (count in t_frames):
                        clip.append(self.resize_frame(frame, params.reso_h, params.reso_w))
                        list_full.append(count)

                else:
                    break

            if len(clip) < params.num_frames and len(clip) > (params.num_frames / 2):
                remaining_num_frames = params.num_frames - len(clip)
                clip = clip + clip[::-1][1:remaining_num_frames + 1]

            return clip

        except:
            print('issue')
            return None

    def resize_frame(self, image, x0, y0):

        try:
            image = self.PIL(image)
            image = image.resize((x0,y0))
            image = trans.functional.to_tensor(image)
        except:
            print('resize Error')
            return None

        return image



if __name__ == '__main__':

    train_dataset = basic_dataloader(shuffle=True, data_percentage=0.2)
    train_dataloader = DataLoader(train_dataset, batch_size=8, \
                                 shuffle=True, num_workers=8, collate_fn=collate_fn_train)

    val_dataset = val_dataloader(shuffle=True, data_percentage=0.2)
    valid_dataloader = DataLoader(val_dataset, batch_size=8, \
                                  shuffle=True, num_workers=8,collate_fn=collate_fn_valid)
    print(f'Step involved: {len(train_dataset) / params.batch_size}')
    t = time.time()

    # for i, clips in enumerate(train_dataloader):
    #     if i % 10 == 0:
    #         print(clips.shape)
    #         # clips = clips.permute(0, 1, 3, 4, 2)
    #         # print(f'Full_clip shape is {clips.shape}')
    #         # print(f'Label is {label}')
    # print(f'Time taken to load data is {time.time() - t}')
    # for i, clips in enumerate(valid_dataloader):
    #     if i % 10 == 0:
    #         print(type(clips[0]))
    #         break
    #         # clips = clips.permute(0, 1, 3, 4, 2)
    #         # print(f'Full_clip shape is {clips.shape}')
    #         # print(f'Label is {label}')
    # print(f'Time taken to load data is {time.time() - t}')
