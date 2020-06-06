import argparse
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
from tqdm import tqdm

# from data_loader import RescaleT, SalObjVideoIterable, ToTensorLab
import models

# from models import U2NET  # full size version 173.6 MB
# from models import U2NETP  # small version u2net 4.7 MB
#from models import U2NETP_short
#from models import JITNET

# import torch.optim as optim

# All threads should propogate quits when they receive the string
# "quit" as an input, and emit the signal downstream

''' A > B
{   np_image:   original numpy image
    image:      processed Torch tensor of image
    id:         id of associated video  }   '''
student_inference_queue = queue.Queue(3)
''' A > C
{   image:     original numpy image
    label?:     original numpy label
    id:         id of associated video  }   '''
orig_image_queue = queue.Queue()
''' B > C
    pred: predicted torch tensor'''
student_result_queue = queue.Queue(3)


# TODO: make abstract or flagify
img_bg = io.imread("data/example_bgs/tokyo.jpg")
img_bg = Image.fromarray(img_bg)
img_bg_resized = None


def paint_output(image_name, pred, orig, d_dir, width=None, height=None):
    # predict = (pred > 0.5).float()
    predict = pred.squeeze().float()
    predict = torch.clamp(predict * 2 - 1, 0, 1)
    predict = predict.cpu().data.numpy()
    del pred

    im = Image.fromarray(predict*255).convert('RGB')
    img_name = image_name.split("/")[-1]

    orig_image_arr = orig
    pred_mask_arr = np.array(im.resize(
        (orig_image_arr.shape[1], orig_image_arr.shape[0]), resample=Image.BILINEAR), dtype=np.float32)
    global img_bg_resized
    if img_bg_resized is None:
        img_bg_resized = np.array(img_bg.resize(
            (orig_image_arr.shape[1], orig_image_arr.shape[0]), resample=Image.BILINEAR))
    inv_mask = 255 - pred_mask_arr
    bg = (inv_mask / 255) * img_bg_resized
    bg = bg.astype(np.uint8)
    pred_img_arr = orig_image_arr * pred_mask_arr / 255
    pred_img_arr = pred_img_arr.astype(np.uint8)
    out = pred_img_arr + bg

    return out


def np_img_to_torch(np_img):
    image = np_img/np.max(np_img)
    tmpImg = np.zeros((image.shape[0], image.shape[1], 3))
    tmpImg[:, :, 0] = (image[:, :, 0]-0.485)/0.229
    tmpImg[:, :, 1] = (image[:, :, 1]-0.456)/0.224
    tmpImg[:, :, 2] = (image[:, :, 2]-0.406)/0.225
    # RGB to BRG
    tmpImg = tmpImg.transpose((2, 0, 1))
    return torch.from_numpy(tmpImg)


def np_img_resize(np_img, width=320, height=320):
    resized_img = Image.fromarray(np_img).convert('RGB')
    resized_img = resized_img.resize(
        (width, height), resample=Image.BILINEAR)
    return np.array(resized_img)


def davis_thread_func():
    davis_path = "data/davis"
    # todo: add other text files
    davis_files = [os.path.join(davis_path, 'ImageSets/480p', txt)
                   for txt in ['train.txt']]
    for f in davis_files:
        imageset_file = open(f, 'r')
        for line in tqdm(imageset_file.readlines()):
            im_anno_pair = line.split(" ")
            vid_id = os.path.join(davis_path, im_anno_pair[0][1:])
            im = io.imread(os.path.join(davis_path, im_anno_pair[0][1:]))
            anno = io.imread(os.path.join(davis_path, im_anno_pair[1][1:]))
            orig_image_queue.put({
                'image': im,
                'id': vid_id,
                'label': anno,
            })

            resized_img = np_img_resize(im)
            tensor = np_img_to_torch(resized_img)
            student_inference_queue.put({
                "np_image": resized_img,
                "image": tensor
            })
    orig_image_queue.put("kill")
    student_inference_queue.put("kill")

def people_thread_func():
    data_path = "data/Supervisely Person Dataset"
    # todo: add other text files
    
    dirpath = [os.path.join(data_path,d,'masks_human') for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path,d))]
    for d in dirpath:
        files = [f for f in os.listdir(d) if os.path.isfile(os.path.join(d,f))]
        for f in files:
            vid_id = os.path.join(d, f)
            image = io.imread(vid_id)
            im = image[:,:int(image.shape[1]/2),:]
            anno = image[:,int(image.shape[1]/2):,:]
            anno = 255*(np.sum((anno[:,:,::-1]-im[:,:,::-1]),axis=2)>0)
            orig_image_queue.put({
                'image': im,
                'id': vid_id,
                'label': anno,
            })
            resized_img = np_img_resize(im)
            tensor = np_img_to_torch(resized_img)
            student_inference_queue.put({
                "np_image": resized_img,
                "image": tensor
            })
    orig_image_queue.put("kill")
    student_inference_queue.put("kill")

def cv2_thread_func(video_name):
    video = cv2.VideoCapture(video_name)
    images_in_flight = []
    try:
        while True:
            succ, image = video.read()
            # image = cv2.resize(image,
            #     (image.shape[1] * 320 // image.shape[0], 320),
            #     interpolation=cv2.INTER_AREA)

            image = image[:, :, ::-1]
            orig_image = image.copy()
            orig_image_queue.put({
                'image': orig_image,
                'id': video_name
            })

            resized_img = np_img_resize(image)
            tensor = np_img_to_torch(resized_img)

            student_inference_queue.put({
                "np_image": resized_img,  # orig_image[:,:,::-1],
                "image": tensor
            })
    except:
        print("CV2 reader hard exit")
        student_inference_queue.put("kill")
        orig_image_queue.put("kill")
    exit()


def paint_thread_func(show=True, keep_video_at=""):
    if show:
        cv2.namedWindow("im")
    vid_out = None
    # TODO: make flag for video saving params
    t = datetime.now()
    total_frames = 0
    while True:
        orig_image = orig_image_queue.get()
        if orig_image == "kill":
            break
        orig_image_np = orig_image['image']
        if not vid_out and keep_video_at:
            vid_out = cv2.VideoWriter(keep_video_at,
                                      cv2.VideoWriter_fourcc(
                                          'M', 'P', '4', 'V'),
                                      # TODO: get fps from cv2 thread message
                                      25, (orig_image_np.shape[1], orig_image_np.shape[0]))
        pred_mask = student_result_queue.get()
        total_frames += 1
        if pred_mask == "kill":
            break
        merged_image = paint_output(
            "", pred_mask, orig_image_np, "")[:, :, ::-1]
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


def score_thread_func(groundtruth, all_classes):
    scores = []
    scores_clamped = []
    cnt = 0
    model = None
    if groundtruth != "label":
        model = models.MODEL_ZOO(groundtruth, all_classes)
    # cv2.namedWindow("scorer debug")
    while True:
        orig_image = orig_image_queue.get()
        if orig_image == "kill":
            break
        if groundtruth=='label':
            orig_label_np = orig_image['label']
            if len(orig_label_np.shape) == 3:
                orig_label_np = orig_label_np[:, :, 0]
        else:
            orig_image_np = orig_image['image']
            data_test = np_img_resize(orig_image_np)
            td1, td2, td3, td4, td5, td6, td7 = model(data_test[:,:,::-1])
            mask = td1.squeeze(0).squeeze(0).detach().float()
            mask = ((mask > 0.5).bool().cpu().numpy() * 255).astype(np.uint8)
            mask = np_img_resize(mask, orig_image_np.shape[1], orig_image_np.shape[0]) > 128
            mask = (mask[:, :, 0] * 255).astype(np.uint8)
            orig_label_np = mask
            del td1, td2, td3, td4, td5, td6, td7
        pred_mask = student_result_queue.get()
        # total_frames += 1
        if pred_mask == "kill":
            break
        pred_mask = ((pred_mask > 0.5).bool().cpu().numpy() * 255).astype(np.uint8)
        pred_mask = np_img_resize(pred_mask, orig_label_np.shape[1], orig_label_np.shape[0]) > 128
        pred_mask = (pred_mask[:, :, 0] * 255).astype(np.uint8)

        union = pred_mask | orig_label_np
        intersection = pred_mask & orig_label_np
        
        if union.sum() == 0:
            score = 1
        else:
            score = intersection.sum() / union.sum()
        if score < 0.1:
            cnt += 1
            print(score)
            print(cnt)
            id = orig_image["id"].split('/')
            print(id[-2] +'_' + id[-1])
        else:
            scores_clamped.append(score)
        scores.append(score)
        print('scores: ' + str(sum(scores) / len(scores)))
    print('scores_clamped: ' + str(sum(scores_clamped) / len(scores_clamped)))


def main():

    parser = argparse.ArgumentParser(
        description='Zoom Matting stream inference pipeline')

    parser.add_argument('--teacher', '-t', default='mrcnn_50',
                        help='name of teacher model: u2net | u2netp | rcnn_101 | mrcnn_50')
    parser.add_argument('--student', '-m', default='jitnet',
                        help='name of student model: u2netp | u2netps | jitnet [not yet impl]')
    parser.add_argument('--headless', '-hl', action='store_true',
                        help='whether to show live feed of inference')
    parser.add_argument('--teacher_mode', '-tm', action='store_true',
                        help='skip student, infer with teacher')
    parser.add_argument('--groundtruth', default='label',
                        help='use label | rcnn_101 | mrcnn_50 prediction as groud truth')
    parser.add_argument(
        '--output', '-o', default = './data/out/jitnet.mp4', help='path to output saved video; none by default')
    parser.add_argument(
        '--input', '-i', default='./data/example_videos/easy/0001.mp4', help='path to input video, 0 for local camera, http://10.1.10.17:8080/video" for IP camera, and dir for video input')
    parser.add_argument(
        '--dataset', default='video', help='run on the video | davis | people dataset'
    )
    parser.add_argument(
        '--hardedge', action='store_true', help='predict image with hard edge (no transparency)'
    )
    parser.add_argument('--score', '-s', action='store_true',
        help='score the model on precomputed targets; always headless')
    parser.add_argument('--all_classes', action='store_true',
        help='include all classes instead of only people segmentation')

    args = parser.parse_args()

    # --------- 1. get image path and name ---------
    teacher_model_name = args.teacher  # 'u2net'#'u2netp'#'rcnn_101'#'mrcnn_50'#
    model_zoo = 'u2net' not in teacher_model_name  # False for u2net/p/short

    student_model_name = args.student  # 'u2net'#'u2netp'#'jitnet'#

    # TODO: make input to script
    # video_path = 0 # for local camera
    # video_path = args.input
    # video_path = "http://10.1.10.17:8080/video" # IP camera
    teacher_model_dir = './saved_models/' + teacher_model_name + '/' + teacher_model_name + '.pth'
    student_model_dir = './saved_models/' + student_model_name + '/' + student_model_name + '.pth'

    # --------- 2. model define ---------
    # Load Teacher
    teacher = None
    if(teacher_model_name == 'u2net'):
        print("...load U2NET---173.6 MB")
        teacher = models.U2NET(3, 1)
    elif(teacher_model_name == 'u2netp'):
        print("...load U2NEP---4.7 MB")
        teacher = models.U2NETP(3, 1)
    elif(model_zoo):
        print("...load MODEL_ZOO: "+teacher_model_dir)
        teacher = models.MODEL_ZOO(teacher_model_name, args.all_classes)

    # Load Student
    if(args.student == 'jitnet'):
        print("...studnet load JITNET")
        student = models.JITNET(3, 1)
    elif(args.student == 'u2net'):
        print("...studnet load U2NET")
        student = models.U2NET(3, 1)
    elif(args.student == 'u2netp'):
        print("...studnet load U2NETP")
        student = models.U2NETP(3, 1)
    elif(args.student == 'u2netp_short'):
        print("...studnet load U2NETP_short")
        student = models.U2NETP_short(3, 1)
    elif(args.student == 'jitnet_side'):
        print("...studnet load JITNET_SIDE")
        student = models.JITNET_SIDE(3, 1)
        
    # Load teacher
    if not model_zoo:
        if torch.cuda.is_available():
            teacher.load_state_dict(torch.load(teacher_model_dir))
            teacher.cuda()
        else:
            teacher.load_state_dict(torch.load(
                teacher_model_dir, map_location=torch.device('cpu')))
        teacher.eval()
    # Load student
    if args.student in ["jitnet", "u2net", "u2netp"]:
        if torch.cuda.is_available():
            student.load_state_dict(torch.load(student_model_dir))
        else:
            student.load_state_dict(torch.load(
                student_model_dir, map_location=torch.device('cpu')))
    if torch.cuda.is_available():
        student.cuda()
    # student.eval()

    # --------- 3. threads setup ---------
    if args.dataset == 'davis':
        producer = threading.Thread(target=davis_thread_func)
    elif args.dataset == 'people':
        producer = threading.Thread(target=people_thread_func)
    elif args.dataset == 'video':
        producer = threading.Thread(target=cv2_thread_func, args=[
            args.input
        ])
    if args.score:
        reducer = threading.Thread(target=score_thread_func, args=(
            args.groundtruth,
            args.all_classes
        ))
    else:
        reducer = threading.Thread(target=paint_thread_func, args=(
            not args.headless, args.output
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
    LOSS_THRESH = 0.001
    delta = DELTA_MIN
    delta_remain = 1
    # cnt = 0
    # --------- 4. inference for each image ---------

    def teacher_infer(data_test):
        if model_zoo:
            data_test = data_test['np_image']
            data_test = data_test[:,:,::-1]
        else:
            data_test = data_test['image'].unsqueeze(0)
            data_test = data_test.type(torch.FloatTensor)
            if torch.cuda.is_available():
                data_test = data_test.cuda()
        td1, td2, td3, td4, td5, td6, td7 = teacher(data_test)
        pred = td1.squeeze(0).squeeze(0).detach().float()
        #pred = torch.clamp(pred * 2 - 1, 0, 1)
        if args.hardedge:
            pred = (pred > 0.5).bool().float()
        del td1, td2, td3, td4, td5, td6, td7
        return pred

    a = datetime.now()
    if args.teacher_mode:
        while True:
            data_test = student_inference_queue.get()
            if data_test == "kill":
                print("Pytorch thread exiting gracefully")
                student_result_queue.put("kill")
                exit()
            pred = teacher_infer(data_test)
            student_result_queue.put(pred)
            del pred
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
            d1, d2, d3, d4, d5, d6, d7 = student(inputs_test)
            # b += (datetime.now() - a).microseconds
            # c += 1
            # print(b/c)

            # normalization
            pred = d1[:, 0, :, :]
            if delta_remain <= 0:
                if torch.isnan(pred).any():
                    print("WARN: PRED NAN")
                    # print(pred)
                    # print(pred.max())
                    # print(pred.min())
                    continue
                # trigger teacher learning
                teacher_pred = teacher_infer(data_test)

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
                        d1, d2, d3, d4, d5, d6, d7 = student(inputs_test)
                        pred = d1[:, 0, :, :]
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
                del teacher_pred
            pred = pred.squeeze(0).detach()
            #pred = torch.clamp(pred * 2 - 1, 0, 1)
            if args.hardedge:
                pred = (pred > 0.5).bool().float()
            student_result_queue.put(pred)
            del d1, d2, d3, d4, d5, d6, d7, pred


if __name__ == "__main__":
    main()
