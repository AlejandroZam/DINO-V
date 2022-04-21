import numpy as np 
import torch.nn as nn
import torch
import torchvision
#from torchsummary import summary
from r3d import r3d_18
from r3d_classifier import r3d_18_classifier
from r3d import mlp
from vivit import ViViT
import parameters_BL as params


# from r2p1d import r2plus1d_18, embedder
# from classifier_r2p1d import r2plus1d_18_classifier
# # from try1_model import r2plus1d_18_changed
# # from dilated_r2plus1d import r2plus1d_18
from torchvision.models.utils import load_state_dict_from_url

def build_r3d_classifier(num_classes = 102, kin_pretrained = False, self_pretrained = True, saved_model_file = None):
    model = r3d_18_classifier(pretrained = kin_pretrained, progress = False)

    model.layer4[0].conv1[0] = nn.Conv3d(256, 512, kernel_size=(3, 3, 3),\
                                stride=(1, 2, 2), padding=(2, 1, 1),dilation = (2,1,1), bias=False)
    model.layer4[0].downsample[0] = nn.Conv3d(256, 512,\
                          kernel_size = (1, 1, 1), stride = (1, 2, 2), bias=False)

    if self_pretrained == True:
        pretrained = torch.load(saved_model_file)
        pretrained_kvpair = pretrained['state_dict']
        # print(pretrained_kvpair)
        # exit()

        model_kvpair = model.state_dict()
        for layer_name, weights in pretrained_kvpair.items():
            if 'module.1.' in layer_name:
                continue
            if 'fc' in layer_name:
                print(layer_name)
                continue
            layer_name = layer_name.replace('module.0.','')
            model_kvpair[layer_name] = weights   
        model.load_state_dict(model_kvpair, strict=True)
        print(f'model {saved_model_file} loaded successsfully!')
    # exit()
    model.fc = nn.Linear(512, num_classes)
    return model 

def load_r3d_classifier(num_classes = 102, saved_model_file = None):
    model = r3d_18_classifier(pretrained = False, progress = False)

    model.layer4[0].conv1[0] = nn.Conv3d(256, 512, kernel_size=(3, 3, 3),\
                                stride=(1, 2, 2), padding=(2, 1, 1),dilation = (2,1,1), bias=False)
    model.layer4[0].downsample[0] = nn.Conv3d(256, 512,\
                          kernel_size = (1, 1, 1), stride = (1, 2, 2), bias=False)

    model.fc = nn.Linear(512, num_classes)

    pretrained = torch.load(saved_model_file)
    pretrained_kvpair = pretrained['state_dict']

    # print(pretrained_kvpair)
    # exit()

    model_kvpair = model.state_dict()
    for layer_name, weights in pretrained_kvpair.items():
        layer_name = layer_name.replace('module.0.','')
        model_kvpair[layer_name] = weights   
    model.load_state_dict(model_kvpair, strict=True)
    print(f'model {saved_model_file} loaded successsfully!')
    return model 

def load_vivit_classifier(num_classes = 102, saved_model_file=None):
    model = ViViT(params.reso_h, 16, num_frames= 16, dim= 768, depth= 12, heads= 12, pool='cls')

    pretrained = torch.load(saved_model_file)
    pretrained_kvpair = pretrained['student']

    # print(pretrained_kvpair)
    # exit()

    model_kvpair = model.state_dict()
    for layer_name, weights in pretrained_kvpair.items():
        if 'module.1' not in layer_name and 'mlp' not in layer_name:
            layer_name = layer_name.replace('module.0.','')
            model_kvpair[layer_name] = weights   
    model.load_state_dict(model_kvpair, strict=True)
    print(f'model {saved_model_file} loaded successsfully!')
    return model

def build_r3d_backbone():
    model = r3d_18(pretrained = False, progress = False)
    
    model.layer4[0].conv1[0] = nn.Conv3d(256, 512, kernel_size=(3, 3, 3),\
                                stride=(1, 2, 2), padding=(2, 1, 1),dilation = (2,1,1), bias=False)
    model.layer4[0].downsample[0] = nn.Conv3d(256, 512,\
                          kernel_size = (1, 1, 1), stride = (1, 2, 2), bias=False)
    return model

def build_r3d_mlp():
    f = build_r3d_backbone()
    g = mlp()
    model = nn.Sequential(f,g)
    return model
    
def load_r3d_mlp(saved_model_file):
    model = build_r3d_mlp()
    pretrained = torch.load(saved_model_file)
    pretrained_kvpair = pretrained['student']
    model_kvpair = model.state_dict()

    for layer_name, weights in pretrained_kvpair.items():
        # if 'module.1' in layer_name: # removing embedder part which is module.1 in the model+embedder
        #     continue
        layer_name = layer_name.replace('module.','')
        model_kvpair[layer_name] = weights  

    model.load_state_dict(model_kvpair, strict=True)
    print(f'{saved_model_file} loaded successfully')
    
    return model 
# # def arch_r50_classifier(num_classes = 101):
# #     model = arch_r50()
# #     model = nn.Sequential(model, nn.Flatten(), nn.Linear(2048, num_classes))
# #     return model 

# def r2plus1d_18_dilatedv2(num_classes = 102, pretrained = False, progress = False):
#     model = r2plus1d_18(pretrained = False, progress = True)
  
#     # model.layer2[0].conv1[0][3] = nn.Conv3d(230, 128, kernel_size=(3, 1, 1),\
#     #                             stride=(1, 1, 1), padding=(2, 0, 0),dilation = (2,1,1), bias=False)
#     # model.layer3[0].conv1[0][3] = nn.Conv3d(460, 256, kernel_size=(3, 1, 1),\
#     #                             stride=(1, 1, 1), padding=(2, 0, 0),dilation = (2,1,1), bias=False)
#     model.layer4[0].conv1[0][3] = nn.Conv3d(921, 512, kernel_size=(3, 1, 1),\
#                                 stride=(1, 1, 1), padding=(2, 0, 0),dilation = (2,1,1), bias=False)

#     # print(model.layer2[0].downsample[0])
#     # exit()
#     # model.layer2[0].downsample[0] = nn.Conv3d(64,128,\
#     #                       kernel_size = (1, 1, 1), stride = (1, 2, 2), bias=False)
#     # model.layer3[0].downsample[0] = nn.Conv3d(128, 256,\
#     #                       kernel_size = (1, 1, 1), stride = (1, 2, 2), bias=False)
#     model.layer4[0].downsample[0] = nn.Conv3d(256, 512,\
#                           kernel_size = (1, 1, 1), stride = (1, 2, 2), bias=False)

    
    
#     if pretrained:
#         state_dict = load_state_dict_from_url('https://download.pytorch.org/models/r2plus1d_18-91a641e6.pth',
#                                                 progress=progress)
#         model.load_state_dict(state_dict)

#     model = nn.Sequential(model, embedder())
#     return model


# def r2plus1d_18_dilated(num_classes = 102, pretrained = False, progress = False):
#     model = torchvision.models.video.r2plus1d_18(pretrained = False, progress = True)
  
#     model.layer2[0].conv1[0][3] = nn.Conv3d(230, 128, kernel_size=(3, 1, 1),\
#                                 stride=(1, 1, 1), padding=(2, 0, 0),dilation = (2,1,1), bias=False)
#     model.layer3[0].conv1[0][3] = nn.Conv3d(460, 256, kernel_size=(3, 1, 1),\
#                                 stride=(1, 1, 1), padding=(2, 0, 0),dilation = (2,1,1), bias=False)
#     model.layer4[0].conv1[0][3] = nn.Conv3d(921, 512, kernel_size=(3, 1, 1),\
#                                 stride=(1, 1, 1), padding=(2, 0, 0),dilation = (2,1,1), bias=False)

#     # print(model.layer2[0].downsample[0])
#     # exit()
#     model.layer2[0].downsample[0] = nn.Conv3d(64,128,\
#                           kernel_size = (1, 1, 1), stride = (1, 2, 2), bias=False)
#     model.layer3[0].downsample[0] = nn.Conv3d(128, 256,\
#                           kernel_size = (1, 1, 1), stride = (1, 2, 2), bias=False)
#     model.layer4[0].downsample[0] = nn.Conv3d(256, 512,\
#                           kernel_size = (1, 1, 1), stride = (1, 2, 2), bias=False)
    
#     if pretrained:
#         state_dict = load_state_dict_from_url('https://download.pytorch.org/models/r2plus1d_18-91a641e6.pth',
#                                                 progress=progress)
#         model.load_state_dict(state_dict)

#     model.fc = nn.Linear(512, num_classes)

#     return model

# def build_r2plus1d_18_classifier(num_classes = 102):
#     model = r2plus1d_18_classifier()
#     model = nn.Sequential(model, nn.Linear(512,num_classes))
#     return model


# def load_r2plus1d_18_dilatedv2(saved_model_file, num_classes = 102):
#     model = r2plus1d_18_classifier()
#     pretrained = torch.load(saved_model_file)
#     pretrained_kvpair = pretrained['state_dict']

#     model_kvpair = model.state_dict()
#     for layer_name, weights in pretrained_kvpair.items():
#         if 'module.1' in layer_name: # removing embedder part which is module.1 in the model+embedder
#             continue
#         layer_name = layer_name.replace('module.0.','')
#         model_kvpair[layer_name] = weights   
#     model.load_state_dict(model_kvpair, strict=True)
#     print(f'{saved_model_file} loaded successfully')
#     model = nn.Sequential(model, nn.Linear(512,num_classes))
#     return model


if __name__ == '__main__':
    # model = arch_r50_classifier(num_classes = 101)
    # model.eval()
    # model.cuda()
    # summary(model, (16, 3, 112, 112))
    input = torch.rand(5, 16, 3, 224, 224)
    # model = build_r3d_classifier(num_classes = 102, saved_model_file = '/home/c3-0/ishan/ss_saved_models/r3d59v2cont2/model_best_e89_loss_15.573.pth')
    # model = build_r3d_classifier(num_classes = 102, saved_model_file = '/home/br087771/saveModel/clusterfit_202/model_50_bestAcc_0.3174.pth')
    model = load_vivit_classifier(102, 'model_14_best_10.818.pth')
    # sftp://ishan@crcv.eecs.ucf.edu/home/c3-0/ishan/ss_saved_models/r3d59v2cont2/model_best_e102_loss_15.418.pth
    # model = torchvision.models.video.r2plus1d_18(pretrained=True, progress=True)

    # saved_model_file = '/home/c3-0/ishan/ss_saved_models/4self_1e-3/model_e108_loss_0.00.pth'
    # model1 = load_r2plus1d_18_dilatedv2(saved_model_file= saved_model_file)
    # model = nn.Sequential(*list(model.children()))
    # print(model)
    # print()
    # model = r2plus1d_18_dilated()
    # print(model)

    # # print()
    model.eval()
    output = model(input)
    print(input.shape)
    print(output.shape)
    # model1.eval()
    # model1.cuda()
    # output = model((input,'s'))

    # print(output[0].shape)
    # summary(model1, (3, 16, 112, 112))
    # summary(model, (1,(3, 16, 112, 112)))

