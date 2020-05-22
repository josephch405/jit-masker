import glob
import os
import queue
import threading
from datetime import datetime

import cv2
# cv2.setNumThreads(5)
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image, ImageChops
from skimage import io, transform
from torch.autograd import Variable
from torchvision import transforms  # , utils

# from data_loader import RescaleT, SalObjVideoIterable, ToTensorLab
from models import U2NET  # full size version 173.6 MB
from models import U2NETP  # small version u2net 4.7 MB
from models import U2NETP_short

# import torch.optim as optim

student_inference_queue = queue.Queue(1)

orig_image_queue = queue.Queue(2)
student_result_queue = queue.Queue(1)


# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/max(ma-mi, 0.001)

    return dn

# TODO: make abstract or flagify
img_bg = io.imread("data/example_bgs/tokyo.jpg")
img_bg = Image.fromarray(img_bg)
img_bg_resized = None
# post_image = None

def paint_output(image_name,pred,orig,d_dir,width=None, height=None):
    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()
    del pred

    im = Image.fromarray(predict_np*255).convert('RGB')
    img_name = image_name.split("/")[-1]
    
    # orig_image_arr = orig.cpu().data.numpy()[0]
    orig_image_arr = orig
    pred_mask_arr = np.array(im.resize((orig_image_arr.shape[1],orig_image_arr.shape[0]),resample=Image.BILINEAR), dtype=np.float32)
    global img_bg_resized
    if img_bg_resized is None:
        img_bg_resized = np.array(img_bg.resize((orig_image_arr.shape[1],orig_image_arr.shape[0]),resample=Image.BILINEAR))
    inv_mask = 255 - pred_mask_arr
    bg = (inv_mask / 255) * img_bg_resized
    bg = bg.astype(np.uint8)
    pred_img_arr = orig_image_arr * pred_mask_arr / 255
    pred_img_arr = pred_img_arr.astype(np.uint8)
    out = pred_img_arr + bg

    return out

def cv2_thread_func(video_name, output_size=320):
    video = cv2.VideoCapture(video_name)
    images_in_flight = []
    while True:
        succ, image = video.read()
        image = image[:,:,::-1]

        orig_image = image.copy()
        orig_image_queue.put(orig_image)

        resized_img = Image.fromarray(image).convert('RGB')
        resized_img = resized_img.resize((output_size,output_size),resample=Image.BILINEAR)
        resized_img = np.array(resized_img)
        
        image = resized_img/np.max(resized_img)
        tmpImg = np.zeros((image.shape[0],image.shape[1],3))
        tmpImg[:,:,0] = (image[:,:,0]-0.485)/0.229
        tmpImg[:,:,1] = (image[:,:,1]-0.456)/0.224
        tmpImg[:,:,2] = (image[:,:,2]-0.406)/0.225

        # RGB to BRG
        tmpImg = tmpImg.transpose((2, 0, 1))

        student_inference_queue.put({
            # "orig_image": orig_image,
            "image": torch.from_numpy(tmpImg)
        })
        # try:
        #     pred_mask = student_result_queue.get_nowait()
        #     # if pred_mask:
        #     orig_image = images_in_flight.pop(0)
        #     merged_image = paint_output("", pred_mask, orig_image, "")
        #     cv2.imshow("im", merged_image[:,:,::-1])
        #     print("time to paint:", datetime.now() - t)
        #     t = datetime.now()
        #     cv2.waitKey(1)
        # except:
        #     # No frame yet
        #     print("no frame")
        #     continue

def paint_thread_func():
    cv2.namedWindow("im")
    t = datetime.now()
    while True:
        orig_image = orig_image_queue.get()
        pred_mask = student_result_queue.get()
        merged_image = paint_output("", pred_mask, orig_image, "")
        cv2.imshow("im", merged_image[:,:,::-1])
        # print("time to paint:", datetime.now() - t)
        t = datetime.now()
        cv2.waitKey(1)

def main():

    # --------- 1. get image path and name ---------
    model_name='u2net'#u2netp

    # TODO: make input to script
    video_path = 0 # for local camera
    # video_path = './data/example_videos/v0.mp4'
    # video_path = "http://10.1.10.17:8080/video" # IP camera
    prediction_dir = './data/out/'
    model_dir = './saved_models/'+ model_name + '/' + model_name + '.pth'

    # --------- 2. dataloader ---------
    # Not needed, we have a worker thread now
    # --------- 3. model define ---------
    if(model_name=='u2net'):
        print("...load U2NET---173.6 MB")
        teacher = U2NET(3,1)
    elif(model_name=='u2netp'):
        print("...load U2NEP---4.7 MB")
        teacher = U2NETP(3,1)
    teacher.load_state_dict(torch.load(model_dir))

    student = U2NETP_short(3, 1)
    if torch.cuda.is_available():
        teacher.cuda()
        student.cuda()
    teacher.eval()
    # student.eval()

    cv_thread = threading.Thread(target=cv2_thread_func, args=(
        video_path, 320
    ))
    paint_thread = threading.Thread(target=paint_thread_func)
    cv_thread.start()
    paint_thread.start()
    
    critereon = nn.BCELoss()
    optimizer = torch.optim.SGD(student.parameters(), lr=0.01, momentum=0.0)
    
    t_loop = datetime.now()
    frame_until_teach = 0
    U_MAX = 8
    DELTA_MIN = 8
    DELTA_MAX = 64
    delta = DELTA_MIN
    delta_remain = 1
    # --------- 4. inference for each image ---------

    while True:
        delta_remain -= 1
        data_test = student_inference_queue.get()
        inputs_test = data_test['image'].unsqueeze(0)
        inputs_test = inputs_test.type(torch.FloatTensor)
        if torch.cuda.is_available():
            inputs_test = inputs_test.cuda()
        d1,d2,d3,d4,d5,d6,d7= student(inputs_test)
        td1,td2,td3,td4,td5,td6,td7= teacher(inputs_test)

        # normalization
        pred = d1[:,0,:,:]
        pred = normPRED(pred)

        if delta_remain <= 0:
            if torch.isnan(pred).any():
                print("WARN: PRED NAN")
                print(pred.max())
                print(pred.min())
            # trigger teacher learning
            teacher_pred = td1[:,0,:,:].detach()
            teacher_pred = normPRED(teacher_pred)

            loss = critereon(pred, teacher_pred)
            print('loss', loss.item())
            if loss.item() != loss.item():
                print("WARN: LOSS NAN")
                print(pred.min())
                print(pred.max())
                print(teacher_pred.min())
                print(teacher_pred.max())
            budget = U_MAX
            if loss.item() > 0.05:
                # Acceptable loss, skip teaching
                while loss.item() > 0.05 and budget > 0:
                    if loss > 0.5:
                        loss /= torch.norm(loss.detach())
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    d1,d2,d3,d4,d5,d6,d7= student(inputs_test)
                    pred = d1[:,0,:,:]
                    pred = normPRED(pred)
                    loss = critereon(pred, teacher_pred)
                    print('loss', loss.item())
                    budget -= 1
            if loss.item() <= 0.05:
                # Loss still bad after training, decrease delay
                delta = min(DELTA_MAX, 2 * delta)
            else:
                delta = max(DELTA_MIN, delta // 2)
            delta_remain = delta
        student_result_queue.put(pred.detach())
        del d1,d2,d3,d4,d5,d6,d7, pred


if __name__ == "__main__":
    main()
