import os
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms#, utils
# import torch.optim as optim

import numpy as np
from PIL import Image, ImageFilter

import glob
import cv2

from data_loader import RescaleT
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from model import U2NET # full size version 173.6 MB
from model import U2NETP # small version u2net 4.7 MB

# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn

def save_output(image_name,pred,d_dir):

    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    im = Image.fromarray(predict_np*255).convert('RGB')
    img_name = image_name.split(os.sep)[-1]
    image = io.imread(image_name)
    imo = im.resize((image.shape[1],image.shape[0]),resample=Image.BILINEAR)

    pb_np = np.array(imo)

    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1,len(bbb)):
        imidx = imidx + "." + bbb[i]

    imo.save(d_dir+imidx+'.png')

def main():

    # --------- 1. get image path and name ---------
    model_name='u2net' #u2netp !!!
    
    image_dir = '/content/input_images' 
    prediction_dir = '/content/output_images'    

    #Best so far
    model_dir = '/content/drive/MyDrive/BEST_u2net.pth'

    if not os.path.exists(prediction_dir):
        os.makedirs(prediction_dir)

    img_name_list = sorted(glob.glob(image_dir + os.sep + '*'))
    print(img_name_list)

    # --------- 2. dataloader ---------
    #1. dataloader
    test_salobj_dataset = SalObjDataset(img_name_list = img_name_list,
                                        lbl_name_list = [],
                                        transform=transforms.Compose([RescaleT(320), #!!! 320
                                                                      ToTensorLab(flag=0)])
                                        )
    test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=32)

    # --------- 3. model define ---------
    if(model_name=='u2net'):
        print("...load U2NET---173.6 MB")
        net = U2NET(3,1)
    elif(model_name=='u2netp'):
        print("...load U2NEP---4.7 MB")
        net = U2NETP(3,1)
    net.load_state_dict(torch.load(model_dir, map_location=torch.device('cpu')))
    # net = torch.load(model_dir)
    
    # for gpu inference    
    if torch.cuda.is_available():
        net.cuda()
    net.eval()

    # --------- 4. inference for each image ---------
    for i_test, data_test in enumerate(test_salobj_dataloader):

        print("inferencing:",img_name_list[i_test].split(os.sep)[-1])

        hires_image = cv2.imread(img_name_list[i_test])

        height, width = hires_image.shape[:2]

        inputs_test = data_test['image']
        inputs_test = inputs_test.type(torch.FloatTensor)

        # # for gpu inference    
        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)

        # for cpu inference    
        # inputs_test = Variable(inputs_test)

        background_image = np.ones((2048, 2048, 3)) * 255
        hires_image = cv2.resize(hires_image, (2048, 2048), cv2.INTER_LANCZOS4)  

        print('inputs_test', torch.max(inputs_test))

        d1,d2,d3,d4,d5,d6,d7= net(inputs_test)

        prediction = d1[0,0,:,:]
        pred = np.uint8(prediction.cpu().detach().numpy() * 255)
        pred = cv2.resize(pred, (min(height * 4, 9192), min(width * 4, 9192)), cv2.INTER_LANCZOS4)  

        _, pred = cv2.threshold(pred,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)


        pred = cv2.resize(pred, (2048, 2048), cv2.INTER_LANCZOS4)  

        mask = np.expand_dims(pred, -1) / 255

        final_image_white_bg = np.uint8((hires_image * mask) + (background_image * (1 - mask)))

        cv2.imwrite(os.path.join(prediction_dir, os.path.splitext(img_name_list[i_test].split(os.sep)[-1])[0] + '.png'), final_image_white_bg)

        del d1,d2,d3,d4,d5,d6,d7

if __name__ == "__main__":
    main()