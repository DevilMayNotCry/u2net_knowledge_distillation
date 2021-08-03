import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.optim as optim

import numpy as np
import os
import cv2
from PIL import Image
from sklearn.model_selection import train_test_split
import cv2
from data_loader import Rescale
from data_loader import RescaleT
from data_loader import RandomCrop
from data_loader import ToTensor
from data_loader import ToTensorLab, ColorJitter
from data_loader import SalObjDataset

from model import U2NET
from model import U2NETP
from model import U2NET10MB

# ------- 1. define loss function --------

bce_loss = nn.BCELoss(size_average=True)

def d0_loss(d0, labels_v):
	loss0 = bce_loss(d0,labels_v)

	return loss0

def sum_loss(output, labels_v):

	loss0 = bce_loss(output[0],labels_v)
	loss1 = bce_loss(output[1],labels_v)
	loss2 = bce_loss(output[2],labels_v)
	loss3 = bce_loss(output[3],labels_v)
	loss4 = bce_loss(output[4],labels_v)
	loss5 = bce_loss(output[5],labels_v)
	loss6 = bce_loss(output[6],labels_v)

	loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6

	return loss

def loss_fn_kd(outputs, labels, teacher_outputs, params):

    alpha = params['alpha']
    T = params['temperature']
    KD_loss = nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1),
                             F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T) + \
              d0_loss(outputs, labels)

    return KD_loss

# ------- 2. set the directory of training dataset --------

data_dir = '/content/'

tra_image_dir = 'input_images/'
tra_label_dir = 'output_images/'

image_ext = '.png'
label_ext = '.png'

saved_model = '/content/drive/MyDrive/BEST_u2net.pth'

epoch_num = 1000
batch_size_train = 6
batch_size_val = 3
train_num = 0
val_num = 0

img_list = sorted(os.listdir(os.path.join(data_dir, tra_image_dir))) 
lbl_list = sorted(os.listdir(os.path.join(data_dir, tra_label_dir)))

total_img_name_list = []
total_lbl_name_list = []

print('difference', list(set(img_list) - set(lbl_list)))

for i, file_path in enumerate(img_list):
    if os.path.basename(file_path) == os.path.basename(lbl_list[i]):
        total_img_name_list.append(os.path.join(data_dir, tra_image_dir, file_path))
        total_lbl_name_list.append(os.path.join(data_dir, tra_label_dir, lbl_list[i]))

print('total_img_name_list', total_img_name_list)


tra_img_name_list, val_img_name_list, tra_lbl_name_list, val_lbl_name_list = train_test_split(total_img_name_list, total_lbl_name_list, test_size=0.1, random_state=7)
        
print("---")
print("train images: ", len(tra_img_name_list))
print("train labels: ", len(tra_lbl_name_list))
print("val images: ", len(val_img_name_list))
print("val labels: ", len(val_lbl_name_list))
print("---")

train_num = len(tra_img_name_list)
val_num = len(val_img_name_list)

salobj_dataset = SalObjDataset(
    img_name_list=tra_img_name_list,
    lbl_name_list=tra_lbl_name_list,
    transform=transforms.Compose([
        RescaleT(320),
        ColorJitter(brightness=(0.9,1.1),contrast=(0.9,1.1),saturation=(0.9,1.1),hue=(-0.01, 0.01)),
        ToTensorLab(flag=0)]))

valobj_dataset = SalObjDataset(
    img_name_list=val_img_name_list,
    lbl_name_list=val_lbl_name_list,
    transform=transforms.Compose([
        RescaleT(320),
        ToTensorLab(flag=0)]))

salobj_dataloader = DataLoader(salobj_dataset, batch_size=batch_size_train, shuffle=True, num_workers=32)
valobj_dataloader = DataLoader(valobj_dataset, batch_size=batch_size_val, shuffle=False, num_workers=32)


# ------- 3. define model --------
# define the net
teacher_net = U2NET(3, 1)
student_net = U2NET10MB(3, 1)
teacher_net.load_state_dict(torch.load(saved_model), strict = True)

if torch.cuda.is_available():
    teacher_net.cuda()
    student_net.cuda()

# ------- 4. define optimizer --------
print("---define optimizer...")
optimizer = optim.Adam(student_net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

# ------- 5. training process --------
print("---start training...")

ite_num = 0
running_loss = 0.0
ite_num4val = 0

val_ite_num = 0
val_running_loss = 0.0
val_ite_num4val = 0

save_freq = 300

for epoch in range(0, epoch_num):
    student_net.train()
    teacher_net.eval()

    for i, data in enumerate(salobj_dataloader):
        ite_num = ite_num + 1
        ite_num4val = ite_num4val + 1

        inputs, labels = data['image'], data['label']

        inputs = inputs.type(torch.FloatTensor)
        labels = labels.type(torch.FloatTensor)

        # wrap them in Variable
        if torch.cuda.is_available():
            inputs_v, labels_v = Variable(inputs.cuda(), requires_grad=False), Variable(labels.cuda(),
                                                                                        requires_grad=False)
        else:
            inputs_v, labels_v = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)

        
        output_batch = student_net(inputs_v)
        with torch.no_grad():
            output_teacher_batch = teacher_net(inputs_v)
    
        labels_v = labels_v.type_as(output_batch[0])

        params = {'alpha': 0.1, 'temperature': 3}
        loss0 = loss_fn_kd(output_batch[0], labels_v, output_teacher_batch[0], params)

        optimizer.zero_grad()
        loss0.backward()
        optimizer.step()

        running_loss += loss0.data

        print("[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f" % (
        epoch + 1, epoch_num, (i + 1) * batch_size_train, train_num, ite_num, running_loss / ite_num4val))

        middle_output = output_batch[0][0][0] * 255
        middle_output = middle_output.cpu().detach().numpy()
        middle_input = inputs.cpu().detach().numpy()[0][:3] * 255
        middle_input = np.moveaxis(middle_input, 0, 2) 
        middle_label = labels.cpu().detach().numpy()[0][0] * 255

        if ite_num % save_freq == 0:

            #save intermediate results cv2
            cv2.imwrite("/content/images_save/output_itr_%d.png" % (ite_num), middle_output) #!!!
            #cv2.imwrite("/content/images_save/label_itr_%d.png" % (ite_num), middle_label) #!!!
            #cv2.imwrite("saved_models/u2net/input_itr_%d.png" % (ite_num), cv2.cvtColor(middle_input, cv2.COLOR_BGR2RGB)) #!!!
            #cv2.imwrite("saved_models/u2net/difference_output_label_itr_%d.png" % (ite_num), abs(middle_output - middle_label)) #!!!

            student_net.eval()
            with torch.no_grad():
                for i, data in enumerate(valobj_dataloader):
                    val_ite_num = val_ite_num + 1
                    val_ite_num4val = val_ite_num4val + 1

                    inputs, labels = data['image'], data['label']

                    inputs = inputs.type(torch.FloatTensor)
                    labels = labels.type(torch.FloatTensor)

                    # wrap them in Variable
                    if torch.cuda.is_available():
                        inputs_v, labels_v = Variable(inputs.cuda(), requires_grad=False), Variable(labels.cuda(),
                                                                                                    requires_grad=False)
                    else:
                        inputs_v, labels_v = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)

                    # forward 
                    val_student = student_net(inputs_v)
                    val_teacher = teacher_net(inputs_v)
                    labels_v = labels_v.type_as(val_student[0])
                    val_loss = loss_fn_kd(val_student[0], labels_v, val_teacher[0], params)

                    # # print statistics
                    val_running_loss += val_loss.data

                    print("[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] val loss: %3f" % (
                    epoch + 1, epoch_num, (i + 1) * batch_size_val, val_num, val_ite_num, val_running_loss / val_ite_num4val))

                    
            
            #save state dictionary
            torch.save(student_net.state_dict(), "/content/models/itr_%d_train.pth" % (ite_num)) #!!!

            running_loss = 0.0
            running_tar_loss = 0.0

            val_running_loss = 0.0
            val_running_tar_loss = 0.0
            student_net.train()  # resume train

            ite_num4val = 0
            val_ite_num4val = 0


