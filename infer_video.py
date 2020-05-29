import glob
import os
import queue
import sys
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
import models

sys.path.append(os.path.realpath('../detectron2'))

#from models import U2NET  # full size version 173.6 MB
#from models import U2NETP  # small version u2net 4.7 MB
#from models import U2NETP_short
#from models import JITNET

# import torch.optim as optim

# All threads should propogate quits when they receive the string
# "quit" as an input, and emit the signal downstream

# A > B
# { in_image:   original numpy image
#   image:      processed Torch tensor of image
#   id:         id of associated video  }
student_inference_queue = queue.Queue(3)
# A > C
# { image?:     original numpy image
#   label?:     original numpy label
#   id:         id of associated video  }
orig_image_queue = queue.Queue()
# B > C
# pred: predicted torch tensor
student_result_queue = queue.Queue(3)


# TODO: make abstract or flagify
img_bg = io.imread("data/example_bgs/tokyo.jpg")
img_bg = Image.fromarray(img_bg)
img_bg_resized = None

def paint_output(image_name,pred,orig,d_dir,width=None, height=None):
    # predict = (pred > 0.5).float()
    # predict = pred.squeeze().float()
    predict = torch.clamp(pred.squeeze().float() * 2 - 0.5, 0, 1)
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

def davis_thread_func():
    davis_path = "data/davis"
    # wip davis_

def cv2_thread_func(video_name, output_size=320):
    video = cv2.VideoCapture(video_name)
    images_in_flight = []
    try:
        while True:
            succ, image = video.read()
            # image = cv2.resize(image,
            #     (image.shape[1] * 320 // image.shape[0], 320),
            #     interpolation=cv2.INTER_AREA)
            
            image = image[:,:,::-1]
            orig_image = image.copy()
            orig_image_queue.put(orig_image)
            
            
            resized_img = Image.fromarray(image).convert('RGB')
            resized_img = resized_img.resize((output_size,output_size),resample=Image.NEAREST)
            resized_img = np.array(resized_img)
            
            image = resized_img/np.max(resized_img)
            tmpImg = np.zeros((image.shape[0],image.shape[1],3))
            tmpImg[:,:,0] = (image[:,:,0]-0.485)/0.229
            tmpImg[:,:,1] = (image[:,:,1]-0.456)/0.224
            tmpImg[:,:,2] = (image[:,:,2]-0.406)/0.225

            # RGB to BRG
            tmpImg = tmpImg.transpose((2, 0, 1))

            student_inference_queue.put({
                "in_image": resized_img, #orig_image[:,:,::-1],
                "image": torch.from_numpy(tmpImg)
            })
    except:
        print("CV2 reader hard exit")
        student_inference_queue.put("kill")
        orig_image_queue.put("kill")
    exit()

def paint_thread_func(show = True, keep_vid = False):
    if show:
        cv2.namedWindow("im")
    vid_out = None
    # TODO: make flag for video saving params
    t = datetime.now()
    total_frames = 0
    while True:
        orig_image = orig_image_queue.get()
        if not vid_out and keep_vid:
            vid_out = cv2.VideoWriter("out.mp4",
                cv2.VideoWriter_fourcc('M', 'P', '4', 'V'),
                # TODO: get fps from cv2 thread message
                25, (orig_image.shape[1], orig_image.shape[0]))
        pred_mask = student_result_queue.get()
        total_frames += 1
        if (orig_image == "kill") or (pred_mask == "kill"):
            print("Drawer exiting gracefully")
            break
        pred_mask = torch.clamp(pred_mask * 3 - 2, 0, 1)
        merged_image = paint_output("", pred_mask, orig_image, "")[:,:,::-1]
        if show:
            cv2.imshow("im", merged_image)
        print("avg time/frame:", (datetime.now() - t) / total_frames)
        if vid_out:
            vid_out.write(merged_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    print("Writer released")
    if show:
        cv2.destroyAllWindows()
    if vid_out:
        vid_out.release()
    exit()

def main():

    # --------- 1. get image path and name ---------
    model_name='mrcnn_50'#'u2net'#'u2netp'#'rcnn_101'#'mrcnn_50'#
    model_zoo = 'u2net' not in model_name # False for u2net/p/short

    # TODO: make input to script
    # video_path = 0 # for local camera
    video_path = './data/example_videos/zoom.mp4'
    # video_path = "http://10.1.10.17:8080/video" # IP camera
    prediction_dir = './data/out/'
    model_dir = './saved_models/'+ model_name + '/' + model_name + '.pth'
    
    show_video = True
    keep_video = False
    teacher_mode = False

    # --------- 2. model define ---------
    teacher = None
    if(model_name=='u2net'):
        print("...load U2NET---173.6 MB")
        teacher = models.U2NET(3,1)
    elif(model_name=='u2netp'):
        print("...load U2NEP---4.7 MB")
        teacher = models.U2NETP(3,1)
    elif(model_zoo):
        print("...load MODEL_ZOO: "+model_name)
        teacher = models.MODEL_ZOO(model_name)

    student = models.U2NETP_short(3, 1) # U2NETP_short # JITNET
    if not model_zoo:
        if torch.cuda.is_available():
            teacher.load_state_dict(torch.load(model_dir))
            teacher.cuda()
        else:
            teacher.load_state_dict(torch.load(model_dir,map_location=torch.device('cpu')))
        teacher.eval()
    if torch.cuda.is_available():
        student.cuda()
    # student.eval()

    # --------- 3. threads setup ---------
    producer = threading.Thread(target=cv2_thread_func, args=(
        video_path, 320
    ))
    reducer = threading.Thread(target=paint_thread_func, args=(
        show_video, keep_video
    ))
    producer.start()
    reducer.start()
    
    critereon = nn.BCELoss(reduction='none')
    optimizer = torch.optim.SGD(student.parameters(), lr=0.01, momentum=0.0)
    
    t_loop = datetime.now()
    frame_until_teach = 0
    U_MAX = 8
    DELTA_MIN = 8
    DELTA_MAX = 64
    LOSS_THRESH = 0.01
    delta = DELTA_MIN
    delta_remain = 1
    # cnt = 0
    # --------- 4. inference for each image ---------

    a = datetime.now()
    if teacher_mode:
        while True:
            data_test= student_inference_queue.get()
            if data_test == "kill":
                print("Pytorch thread exiting gracefully")
                student_result_queue.put("kill")
                exit()
            if model_zoo:
                data_test = data_test['in_image']
            else:
                data_test = data_test['image'].unsqueeze(0)
                data_test = data_test.type(torch.FloatTensor)
                if torch.cuda.is_available():
                    data_test = data_test.cuda()
            td1,td2,td3,td4,td5,td6,td7= teacher(data_test)
            pred = td1.squeeze(0).squeeze(0)
            student_result_queue.put(pred.detach())
            del td1,td2,td3,td4,td5,td6,td7, pred
            # cnt+=1
            # if cnt == 100:
            #     cnt = 0
            #     print('time/100 frames', datetime.now() - a)
            #     a = datetime.now()

    # b = 0
    # c = 0
    else:
        while True:
            delta_remain -= 1
            data_test = student_inference_queue.get()
            if data_test == "kill":
                print("Pytorch thread exiting gracefully")
                student_result_queue.put("kill")
                exit()
            inputs_test = data_test['image'].unsqueeze(0)
            inputs_test = inputs_test.type(torch.FloatTensor)
            if torch.cuda.is_available():
                inputs_test = inputs_test.cuda()
            # a = datetime.now()
            # print(inputs_test)
            d1,d2,d3,d4,d5,d6,d7= student(inputs_test)
            # b += (datetime.now() - a).microseconds
            # c += 1
            # print(b/c)
    
            # normalization
            pred = d1[:,0,:,:]
            if delta_remain <= 0:
                if torch.isnan(pred).any():
                    print("WARN: PRED NAN")
                    # print(pred)
                    # print(pred.max())
                    # print(pred.min())
                    continue
                # trigger teacher learning
                teach_inputs_test = inputs_test
                if model_zoo:
                    teach_inputs_test = data_test['in_image'].copy()
                td1,td2,td3,td4,td5,td6,td7= teacher(teach_inputs_test)
                if model_zoo:
                    teacher_pred = td1.detach().squeeze(0).squeeze(0).float()
                else:
                    teacher_pred = td1[:,0,:,:].detach()

                loss = critereon(pred, teacher_pred)
                loss *= ((teacher_pred * 5) + 1) / 6
                loss = loss.mean()
                budget = U_MAX
                # cnt+=1
                # if cnt == 10:
                #     cnt = 0
                #     print('loss', loss.item())
                #     print('time', datetime.now() - a)
                #     a = datetime.now()
                        
                if loss.item() > LOSS_THRESH:
                    # Acceptable loss, skip teaching
                    while loss.item() > LOSS_THRESH and budget > 0:
                        if loss > 0.5:
                            loss /= torch.norm(loss.detach())
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        d1,d2,d3,d4,d5,d6,d7= student(inputs_test)
                        pred = d1[:,0,:,:]
                        loss = critereon(pred, teacher_pred)
                        loss *= ((teacher_pred * 6) + 1) / 6
                        loss = loss.mean()
                        budget -= 1
                if loss.item() <= LOSS_THRESH:
                    # Loss still bad after training, decrease delay
                    delta = min(DELTA_MAX, 2 * delta)
                else:
                    delta = max(DELTA_MIN, delta // 2)
                delta_remain = delta
                del td1,td2,td3,td4,td5,td6,td7, teacher_pred
            student_result_queue.put(pred.squeeze(0).detach())
            del d1,d2,d3,d4,d5,d6,d7, pred


if __name__ == "__main__":
    main()
