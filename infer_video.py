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

import cv2
import numpy as np
from PIL import Image
from PIL import ImageChops
import glob

from data_loader import RescaleT
from data_loader import ToTensorLab
from data_loader import SalObjVideoDataset

from models import U2NET # full size version 173.6 MB
from models import U2NETP # small version u2net 4.7 MB

# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn

# TODO: make abstract or flagify
img_bg = io.imread("data/example_bgs/tokyo.jpg")
img_bg = Image.fromarray(img_bg)

def save_output(image_name,pred,orig,d_dir,width=None, height=None):

    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    im = Image.fromarray(predict_np*255).convert('RGB')
    img_name = image_name.split("/")[-1]
    
    image = orig
    imo = im.resize((image.shape[1],image.shape[0]),resample=Image.BICUBIC)
    img_bg_local = img_bg.resize((image.shape[1],image.shape[0]),resample=Image.BICUBIC)
    # if not width and not height:
    #     image = io.imread(image_name)
    #     imo = im.resize((image.shape[1],image.shape[0]),resample=Image.BICUBIC)
    #     # TODO: maybe make optional
    # else:
    #     imo = im.resize((width, height),resample=Image.BICUBIC)
    inv_mask = ImageChops.invert(imo)
    bg = ImageChops.multiply(inv_mask, img_bg_local)
    imo = ImageChops.multiply(Image.fromarray(image), imo)
    imo = ImageChops.add(imo, bg)

    pb_np = np.array(imo)

    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1,len(bbb)):
        imidx = imidx + "." + bbb[i]
    cv2.namedWindow("im")
    cv2.imshow("im", np.array(imo)[:,:,::-1])
    cv2.waitKey(0)
    # imo.save(d_dir+imidx+'.jpg')

def main():

    # --------- 1. get image path and name ---------
    model_name='u2netp'#u2netp

    # TODO: make input to script
    video_path = './data/example_videos/v0.mp4'
    prediction_dir = './data/out/'
    model_dir = './saved_models/'+ model_name + '/' + model_name + '.pth'

    # --------- 2. dataloader ---------
    #1. dataloader
    test_salobj_dataset = SalObjVideoDataset(video_path,
                                        transform=transforms.Compose([RescaleT(320),
                                                                      ToTensorLab(flag=0)])
                                        )
    # batch_size = 1
    # test_salobj_dataloader = DataLoader(test_salobj_dataset,
    #                                     batch_size=batch_size,
    #                                     shuffle=False,
    #                                     num_workers=1)

    # --------- 3. model define ---------
    if(model_name=='u2net'):
        print("...load U2NET---173.6 MB")
        net = U2NET(3,1)
    elif(model_name=='u2netp'):
        print("...load U2NEP---4.7 MB")
        net = U2NETP(3,1)
    net.load_state_dict(torch.load(model_dir))
    if torch.cuda.is_available():
        net.cuda()
    net.eval()

    # --------- 4. inference for each image ---------
    for i_test, data_test in enumerate(test_salobj_dataset):

        print("inferencing frame:", i_test)
        # print("dl:", datetime.now()-a)
        inputs_test = data_test['image'].unsqueeze(0)
        inputs_test = inputs_test.type(torch.FloatTensor)
        if torch.cuda.is_available():
            inputs_test = inputs_test.cuda()

        d1,d2,d3,d4,d5,d6,d7= net(inputs_test)

        # normalization
        pred = d1[:,0,:,:]
        pred = normPRED(pred)

        # save results to test_results folder
        # TODO: dynamically remember input sizes somehow, hardcoded for now
        # for j in range(pred.shape[0]):
        save_output("out" + str(i_test) + ".jpg",pred,data_test['orig_image'],prediction_dir,1920, 1080)
        # TODO: read dimensions from input

        del d1,d2,d3,d4,d5,d6,d7

if __name__ == "__main__":
    main()
