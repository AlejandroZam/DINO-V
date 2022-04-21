'''
# TO Run this script:
#Turing:
# sbatch -c8 --mem-per-cpu=4G -C gmem11 --gres=gpu:turing:1 --wrap="python train_BL2.py --run_id="" --saved_model="" --restart" --job-name="" --output=".out" --exclude c1-2 --time 4-0 -p preempt --requeue --open-mode=append --qos preempt
#Pascal:
# sbatch -c8 --mem-per-cpu=4G --gres=gpu:pascal:1 --wrap="python train_BL2.py --run_id="" --saved_model="" --restart" --job-name="" --output=".out" --exclude c1-2 --time 4-0 -p preempt --requeue --open-mode=append --qos preempt
# sbatch -c16 --mem-per-cpu=4G --gres=gpu:pascal:1 --wrap="python train_BL2.py --run_id="r3d93e277_50p_ft" --saved_model="sftp://ishan@crcv.eecs.ucf.edu/home/c3-0/ishan/ss_saved_models/r3d93/model_best_e277_loss_9.5951.pth" --restart" --job-name="r3d93e277_50p_ft" --output="r3d93e277_50p_ft.out" --exclude c1-2 --time 4-0 -p gpu

# sbatch -c16 --mem-per-cpu=4G --gres=gpu:pascal:1 --wrap="python train_BL2.py --run_id="r3d93e277_50p_ftV2" --saved_model="sftp://ishan@crcv.eecs.ucf.edu/home/c3-0/ishan/ss_saved_models/r3d93/model_best_e277_loss_9.5951.pth" --restart" --job-name="r3d93e277_50p_ftV2" --output="r3d93e277_50p_ftV2.out" --exclude c1-2 --time 4-0 -p preempt --requeue --open-mode=append --qos preempt

# sbatch -c16 --mem-per-cpu=4G --gres=gpu:pascal:2 --wrap="python train_BL2.py --run_id="ftry3_r3d93e277_32f_112" --saved_model="sftp://ishan@crcv.eecs.ucf.edu/home/c3-0/ishan/ss_saved_models/r3d93/model_best_e277_loss_9.5951.pth" --restart" --job-name="ftry3_r3d93e277_32f_112" --output="ftry3_r3d93e277_32f_112.out" --exclude c1-2 --time 4-0 -p gpu
'''
# sbatch -c16 --mem-per-cpu=4G --gres=gpu:pascal:1 --wrap="python train_BL2.py --run_id="r3d_cvrl_e151" --saved_model="sftp://ishan@crcv.eecs.ucf.edu/home/c3-0/ishan/ss_saved_models/r3d_cvrl/model_best_e151_loss_0.0425.pth" --restart" --job-name="r3d_cvrl_e151" --output="r3d_cvrl_e151.out" --exclude c1-2 --time 4-0 -p gpu 

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import *
import time
import os
import numpy as np
from model import build_r3d_classifier, load_r3d_classifier, load_vivit_classifier
import parameters_BL as params
import config as cfg
#from dl_ft_nov import *
from dl_20p_ft import *
import sys, traceback
from sklearn.metrics import precision_recall_fscore_support, average_precision_score
from tensorboardX import SummaryWriter
import cv2
from torch.utils.data import DataLoader
import math
import argparse
import itertools

from keras.utils import to_categorical

#if torch.cuda.is_available(): 
    #torch.set_default_tensor_type('torch.cuda.FloatTensor')
torch.backends.cudnn.benchmark = True

def train_epoch(run_id, epoch, data_loader, model, criterion, optimizer, writer, use_cuda,learning_rate2):
    print('train at epoch {}'.format(epoch))
    for param_group in optimizer.param_groups:
        param_group['lr']=learning_rate2
        writer.add_scalar('Learning Rate', learning_rate2, epoch)  

        print("Learning rate is: {}".format(param_group['lr']))
  
    losses, weighted_losses = [], []
    loss_mini_batch = 0
    # optimizer.zero_grad()

    model.train()

    for i, (inputs, label, vid_path) in enumerate(data_loader):
        # print(f'label is {label}')
        # inputs = inputs.permute(0,4,1,2,3)
        # print(inputs.shape)
        optimizer.zero_grad()

        inputs = inputs.permute(0,2,1,3,4)
        # print(inputs.shape)
        if use_cuda:
            inputs = inputs.cuda()
            label = torch.from_numpy(np.asarray(label)).cuda()
        output = model(inputs)
        # print(f'Output shape is {output.shape}')
        # print(f'Output[0] is {output[0]}')

        # print(f'inputs shape is {inputs.shape}')
        # print(f'outputs shape is {output.shape}')
        # print(f'label shape is {label.shape}')
        # print(f'label[0] is {label[0]}')
        # exit()
    
        loss = criterion(output,label)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if i % 24 == 0: 
            print(f'Training Epoch {epoch}, Batch {i}, Loss: {np.mean(losses) :.5f}')
        
    print('Training Epoch: %d, Loss: %.4f' % (epoch,  np.mean(losses)))
    writer.add_scalar('Training Loss', np.mean(losses), epoch)

    del loss, inputs, output, label

    return model, np.mean(losses)

def val_epoch(run_id, epoch,mode, skip, hflip, cropping_fac, pred_dict,label_dict, data_loader, model, criterion, writer, use_cuda):
    print(f'validation at epoch {epoch} - mode {mode} - skip {skip} - hflip {hflip} - cropping_fac {cropping_fac}')
    
    model.eval()

    losses = []
    predictions, ground_truth = [], []
    vid_paths = []

    for i, (inputs, label, vid_path, _) in enumerate(data_loader):
        vid_paths.extend(vid_path)
        ground_truth.extend(label)
        # inputs = inputs.permute(0,4,1,2,3)
        if len(inputs.shape) != 1:

            inputs = inputs.permute(0, 2, 1, 3, 4)

            # print(label)
            if use_cuda:
                inputs = inputs.cuda()
                label = torch.from_numpy(np.asarray(label)).cuda()
            # print(label)

        
            with torch.no_grad():

                output = model(inputs)
                loss = criterion(output,label)

            losses.append(loss.item())


            predictions.extend(nn.functional.softmax(output, dim = 1).cpu().data.numpy())
            # print(len(predictions))


            if i+1 % 45 == 0:
                print("Validation Epoch ", epoch , "mode", mode, "skip", skip, "hflip", hflip, " Batch ", i, "- Loss : ", np.mean(losses))
        
    del inputs, output, label, loss 

    ground_truth = np.asarray(ground_truth)
    pred_array = np.flip(np.argsort(predictions,axis=1),axis=1) # Prediction with the most confidence is the first element here
    c_pred = pred_array[:,0] #np.argmax(predictions,axis=1).reshape(len(predictions))

    for entry in range(len(vid_paths)):
        if str(vid_paths[entry].split('/')[-1]) not in pred_dict.keys():
            pred_dict[str(vid_paths[entry].split('/')[-1])] = []
            pred_dict[str(vid_paths[entry].split('/')[-1])].append(predictions[entry])

        else:
            # print('yes')
            pred_dict[str(vid_paths[entry].split('/')[-1])].append(predictions[entry])

    for entry in range(len(vid_paths)):
        if str(vid_paths[entry].split('/')[-1]) not in label_dict.keys():
            label_dict[str(vid_paths[entry].split('/')[-1])]= ground_truth[entry]

    print_pred_array = []

    # for entry in range(pred_array.shape[0]):
    #     temp = ''
    #     for i in range(5):
    #         temp += str(int(pred_array[entry][i]))+' '
    #     print_pred_array.append(temp)
    # print(f'check {print_pred_array[0]}')
    # results = open('Submission1.txt','w')
    # for entry in range(len(vid_paths)):
    #     content = str(vid_paths[entry].split('/')[-1] + ' ' + print_pred_array[entry])[:-1]+'\n'
    #     results.write(content)
    # results.close()
    
    correct_count = np.sum(c_pred==ground_truth)
    accuracy = float(correct_count)/len(c_pred)
    
    # print(f'Correct Count is {correct_count}')
    print(f'Epoch {epoch}, mode {mode}, skip {skip}, hflip {hflip}, cropping_fac {cropping_fac}, Accuracy: {accuracy*100 :.3f}')
    return pred_dict, label_dict, accuracy, np.mean(losses)
    
def train_classifier(run_id, restart, saved_model):
    use_cuda = True
    best_score = 0
    writer = SummaryWriter(os.path.join(cfg.logs, str(run_id)))

    save_dir = os.path.join(cfg.saved_models_dir, run_id)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # saved_model_file = '/home/c3-0/ishan/ss_saved_models/r3d50cont2/model_best_e198_loss_4.2249.pth'

    # model =  build_r3d_classifier_early(num_classes = 102, self_pretrained = True, saved_model_file = saved_model_file)
    # model = build_ken_r2plus1d_classifier(num_classes = 102)
    
    # model = build_r2plus1d_prp(num_classes = 102)


    # saved_model_file = '/home/c3-0/ishan/ss_saved_models/1rpd/model_best_e126_loss_59.149.pth'
    # model = build_r21d_prp_modified_classifier(self_pretrained = True, saved_model_file = saved_model_file)
    # model = build_r21d_prp_modified_classifier(self_pretrained = False, saved_model_file = None)
    
    # model = build_c3d_classifier(num_classes = 102)
    
    # saved_model_file = '/home/c3-0/ishan/ss_saved_models/1c3dcont2/model_best_e284_loss_22.352.pth'
    # saved_model_file = '/home/c3-0/ishan/ss_saved_models/1c3dcont/model_best_e152_loss_59.355.pth'
    # saved_model_file = '/home/c3-0/ishan/ss_saved_models/1c3dcont/model_best_e225_loss_29.895.pth'
    # saved_model_file = '/home/c3-0/ishan/ss_saved_models/1c3dcont2/model_best_e396_loss_24.617.pth'
    # saved_model_file = '/home/c3-0/ishan/ss_saved_models/r3d57cont/model_best_e427_loss_30.420.pth'
    # saved_model_file = '/home/c3-0/ishan/ss_saved_models/r3d57cont/model_best_e226_loss_26.205.pth'
    # saved_model_file = '/home/c3-0/ishan/ss_saved_models/r3d57cont/model_best_e158_loss_69.905.pth'
    # saved_model_file = '/home/c3-0/ishan/ss_saved_models/dummy_r3d31_repeat/model_best_e196_loss_15.729.pth'
    # saved_model_file = '/home/c3-0/ishan/ss_saved_models/r3d59v2cont2/model_best_e53_loss_18.625.pth'
    # saved_model_file = saved_model[30:] #sftp://ishan@crcv.eecs.ucf.edu/home/c3-0/ishan/ss_saved_models/r3d59v2cont2/model_e125_loss_15.102.pth
    #'/home/c3-0/ishan/ss_saved_models/r3d59v2cont2correct/model_best_e31_loss_19.029.pth'
    # saved_model_file = '/home/c3-0/ishan/ss_saved_models/dummy_r3d58/model_best_e146_loss_0.2693.pth'
    '''if restart:
        saved_model_file = save_dir + '/model_temp.pth'
        
        if os.path.exists(saved_model_file):
            model = load_r3d_classifier(saved_model_file= saved_model_file)
            epoch0 = torch.load(saved_model_file)['epoch']
        else:
            print(f'No such model exists: {saved_model_file} :(')
            if not (saved_model == None or len(saved_model) == 0):
                print(f'Trying to load {saved_model[30:]}')
                model = build_r3d_classifier(saved_model_file = saved_model[30:], num_classes = params.num_classes)
            else:
                print(f'It`s a baseline experiment!')
                model = build_r3d_classifier(self_pretrained = False, saved_model_file = None, num_classes = params.num_classes) 
            epoch0 = 0

    else:
        epoch0 = 0
        model = build_r3d_classifier(saved_model_file = saved_model, num_classes = params.num_classes)'''
    
    epoch0 = 0
    model = load_vivit_classifier(102, '/home/cap6412.student1/DINO/train_models/model_87_best_3.9215.pth')

    # saved_model_file = '/home/c3-0/ishan/ss_saved_models/r3d58/model_best_e154_loss_0.7567.pth'
    learning_rate1 = params.learning_rate
    
    # temp = list(np.linspace(0,1, 10) + 1e-9) + [1 for i in range(50)] + [0.1 for i in range(100)]

    # lr_array = l*np.asarray(temp)
    learning_rate2 = learning_rate1 

    
    # print(lr_array[:50])
    criterion= nn.CrossEntropyLoss()
    # criterion = torch.hub.load('adeelh/pytorch-multi-class-focal-loss',model='focal_loss',gamma=1,reduction='mean',device='cpu',dtype=torch.float32,force_reload=False)

    if torch.cuda.device_count()>1:
        print(f'Multiple GPUS found!')
        model=nn.DataParallel(model)
        model.cuda()
        criterion.cuda()
    else:
        print('Only 1 GPU is available')
        model.cuda()
        criterion.cuda()

    optimizer = optim.Adam(model.parameters(),lr=params.learning_rate)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.80)

    train_dataset = baseline_dataloader_train_strong(shuffle = False, data_percentage = params.data_percentage)
    # train_dataset = baseline_dataloader(shuffle = False, data_percentage = 0.1)

    train_dataloader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True, num_workers=params.num_workers, collate_fn=collate_fn_train)
    print(f'Train dataset length: {len(train_dataset)}')
    print(f'Train dataset steps per epoch: {len(train_dataset)/params.batch_size}')
    # val_array = list(range(250))#[0,1,25,50,100,150,175] + list(range(200,255))
    # val_array = [0,9,60] + [90, 93, 96, 99] + [125+ x for x in range(100)]
    # val_array = [0,10,20, 25] + [12+ x for x in range(100)]
    val_array = [0] + [5*x for x in range(1, 11)] + [50+2*x for x in range(1, 126)]
    # val_array = []

    modes = list(range(params.num_modes))
    skip = list(range(1,params.num_skips+1))
    hflip = params.hflip#list(range(2))
    cropping_fac1 = params.cropping_fac1#[0.7,0.85,0.8,0.75]
    print(f'Num modes {len(modes)}')
    print(f'Num skips {skip}')
    print(f'Cropping fac {cropping_fac1}')
    modes, skip,hflip, cropping_fac =  list(zip(*itertools.product(modes,skip,hflip,cropping_fac1)))
    accuracy = 0
    lr_flag1 = 0


    
    for epoch in range(epoch0, params.num_epochs):
        if epoch < params.warmup and lr_flag1 ==0:
            learning_rate2 = params.warmup_array[epoch]*params.learning_rate


        print(f'Epoch {epoch} started')
        start=time.time()
        try:
            model, train_loss = train_epoch(run_id, epoch, train_dataloader, model, criterion, optimizer, writer, use_cuda,learning_rate2)
           
            if train_loss < 0.8 and lr_flag1 ==0:
                lr_flag1 =1 
                learning_rate2 = learning_rate1/2
                print(f'Dropping learning rate to {learning_rate2} for epoch')

            if train_loss < 0.4 and lr_flag1 ==1:
                lr_flag1 = 2  
                learning_rate2 = learning_rate1/10
                print(f'Dropping learning rate to {learning_rate2} for epoch')

            if train_loss < 0.1 and lr_flag1 ==2:
                lr_flag1 = 3
                learning_rate2 = learning_rate1/20
                print(f'Dropping learning rate to {learning_rate2} for epoch')

            '''if train_loss < 0.05 and lr_flag1 ==3:
                lr_flag1 = 4
                learning_rate2 = learning_rate1/100
                print(f'Dropping learning rate to {learning_rate2} for epoch')

            if train_loss < 0.005 and lr_flag1 ==4:
                lr_flag1 = 5
                learning_rate2 = learning_rate1/1000
                print(f'Dropping learning rate to {learning_rate2} for epoch')'''

            if epoch in val_array:
                pred_dict = {}
                label_dict = {}
                val_losses = []

                for val_iter in range(len(modes)-1):
                    try:
                        validation_dataset = multi_baseline_dataloader_val_strong(shuffle = True, data_percentage = params.data_percentage,\
                            mode = modes[val_iter], skip = skip[val_iter], hflip= hflip[val_iter], \
                            cropping_factor= cropping_fac[val_iter])
                        validation_dataloader = DataLoader(validation_dataset, batch_size=params.v_batch_size, shuffle=True, num_workers=params.num_workers, collate_fn=collate_fn2)
                        if val_iter ==0:
                            print(f'Validation dataset length: {len(validation_dataset)}')
                            print(f'Validation dataset steps per epoch: {len(validation_dataset)/params.v_batch_size}')    
                        pred_dict, label_dict, accuracy, loss = val_epoch(run_id, epoch,modes[val_iter],skip[val_iter],hflip[val_iter],cropping_fac[val_iter], \
                            pred_dict, label_dict, validation_dataloader, model, criterion, writer, use_cuda)
                        val_losses.append(loss)

                        predictions1 = np.zeros((len(list(pred_dict.keys())),params.num_classes))
                        ground_truth1 = []
                        entry = 0
                        for key in pred_dict.keys():
                            predictions1[entry] = np.mean(pred_dict[key], axis =0)
                            entry+=1

                        for key in label_dict.keys():
                            ground_truth1.append(label_dict[key])
                        
                        pred_array1 = np.flip(np.argsort(predictions1,axis=1),axis=1) # Prediction with the most confidence is the first element here
                        c_pred1 = pred_array1[:,0]

                        correct_count1 = np.sum(c_pred1==ground_truth1)
                        accuracy11 = float(correct_count1)/len(c_pred1)

                        
                        print(f'Running Avg Accuracy is for epoch {epoch}, mode {modes[val_iter]}, skip {skip[val_iter]}, hflip {hflip[val_iter]}, cropping_fac {cropping_fac[val_iter]} is {accuracy11*100 :.3f}% ')  
                    except:
                        print(f'Failed epoch {epoch}, mode {modes[val_iter]}, skip {skip[val_iter]}, hflip {hflip[val_iter]}, cropping_fac {cropping_fac[val_iter]} is {accuracy11*100 :.3f}% ')  

                val_loss = np.mean(val_losses)
                predictions = np.zeros((len(list(pred_dict.keys())),params.num_classes))
                ground_truth = []
                entry = 0
                for key in pred_dict.keys():
                    predictions[entry] = np.mean(pred_dict[key], axis =0)
                    entry+=1

                for key in label_dict.keys():
                    ground_truth.append(label_dict[key])
                
                pred_array = np.flip(np.argsort(predictions,axis=1),axis=1) # Prediction with the most confidence is the first element here
                c_pred = pred_array[:,0]

                correct_count = np.sum(c_pred==ground_truth)
                accuracy1 = float(correct_count)/len(c_pred)
                print(f'Correct Count is {correct_count} out of {len(c_pred)}')
                writer.add_scalar('Validation Loss', np.mean(val_loss), epoch)
                writer.add_scalar('Validation Accuracy', np.mean(accuracy1), epoch)
                
                print(f'Overall Accuracy is for epoch {epoch} is {accuracy1*100 :.3f}% ')
                # file_name = f'RunID_{run_id}_Acc_{accuracy1*100 :.3f}_cf_{len(cropping_fac1)}_m_{params.num_modes}_s_{params.num_skips}.pkl'     
                # pickle.dump(pred_dict, open(file_name,'wb'))
                accuracy = accuracy1

            if accuracy > best_score:
                print('++++++++++++++++++++++++++++++')
                print(f'Epoch {epoch} is the best model till now for {run_id}!')
                print('++++++++++++++++++++++++++++++')
                save_dir = os.path.join(cfg.saved_models_dir, run_id)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                save_file_path = os.path.join(save_dir, 'model_{}_bestAcc_{}.pth'.format(epoch, str(accuracy)[:6]))
                states = {
                    'epoch': epoch + 1,
                    # 'arch': params.arch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                torch.save(states, save_file_path)
                best_score = accuracy
            # else:
            if epoch%5 == 0:
                save_dir = os.path.join(cfg.saved_models_dir, run_id)
                save_file_path = os.path.join(save_dir, 'model_{}.pth'.format(epoch))
                states = {
                    'epoch': epoch + 1,
                    # 'arch': params.arch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                torch.save(states, save_file_path)
            save_dir = os.path.join(cfg.saved_models_dir, run_id)
            save_file_path = os.path.join(save_dir, 'model_temp.pth')
            states = {
                'epoch': epoch + 1,
                # 'arch': params.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(states, save_file_path)
            # scheduler.step()
            # elif epoch % 20 == 0:
            #     save_dir = os.path.join(cfg.saved_models_dir, run_id)
            #     save_file_path = os.path.join(save_dir, 'model_{}_Acc_{}_F1_{}.pth'.format(epoch, str(accuracy)[:6],str(f1_score)[:6]))
            #     states = {
            #         'epoch': epoch + 1,
            #         # 'arch': params.arch,
            #         'state_dict': model.state_dict(),
            #         'optimizer': optimizer.state_dict(),
            #     }
            #     torch.save(states, save_file_path)

            # scheduler.step()
        except:
            print("Epoch ", epoch, " failed")
            print('-'*60)
            traceback.print_exc(file=sys.stdout)
            print('-'*60)
            continue
        taken = time.time()-start
        print(f'Time taken for Epoch-{epoch} is {taken}')
        print()
        train_dataset = baseline_dataloader_train_strong(shuffle = False, data_percentage = params.data_percentage)
        train_dataloader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True, num_workers=params.num_workers, collate_fn=collate_fn_train)
        print(f'Train dataset length: {len(train_dataset)}')
        print(f'Train dataset steps per epoch: {len(train_dataset)/params.batch_size}')


        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to train baseline')

    parser.add_argument("--run_id", dest='run_id', type=str, required=False, default= "dummy_ft",
                        help='run_id')
    parser.add_argument("--restart", action='store_true')
    parser.add_argument("--saved_model", dest='saved_model', type=str, required=False, default= None,
                        help='run_id')

    # print()
    # print('Repeating r3d57, Optimizer grad inside each iteration')
    # print()


    args = parser.parse_args()
    print(f'Restart {args.restart}', flush=True)
    print(f'torch.version.cuda: {torch.version.cuda}', flush=True)
    print(f'is available: {torch.cuda.is_available()}')
    print(f'backend: {torch.backends.cudnn.enabled}')

    run_id = args.run_id
    saved_model = args.saved_model

    train_classifier(str(run_id), args.restart, saved_model)
