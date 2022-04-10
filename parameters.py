import numpy as np

num_frames = 16
num_workers = 11#8
batch_size = 1#24
learning_rate = 1e-4#restarting1e-3#1e-4#1e-5
num_epochs = 300
data_percentage = 1.0
v_batch_size = 32#80

fix_skip = 2
num_modes = 10
num_skips = 1
hflip = [0] #list(range(2))
cropping_fac1 = [0.8] #[0.7,0.85,0.8,0.75]

reso_h = 224
reso_w = 224

ori_reso_h = 240
ori_reso_w = 320
sr_ratio = 4


warmup_array = list(np.linspace(0.01,1, 10) + 1e-9)
warmup = len(warmup_array)

num_classes = 102
temperature = 3.0
alpha = 0.1