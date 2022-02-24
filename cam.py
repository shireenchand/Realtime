import cv2
from predict import Predict
from pytorch_i3d import InceptionI3d
import torch
import torch.nn as nn
from multiprocessing import Pool
import os
import multiprocessing
import time

i3d = InceptionI3d(400, in_channels=3)
i3d.load_state_dict(torch.load('rgb_imagenet.pt',map_location=torch.device('cpu')))
i3d.replace_logits(2000)
i3d.load_state_dict(torch.load("nslt_2000_065846_0.447803.pt",map_location=torch.device('cpu')))
# i3d.cuda()
i3d = nn.DataParallel(i3d)
i3d.eval()

vid = cv2.VideoCapture(0)
cnt = 0
frames = []
while(True):
    ret,frame = vid.read()
    cnt += 1
    cv2.imshow("frame",frame)
    frames.append(frame)
    if cnt == 64:
        print("64 Frames")
        pred = Predict(frames,i3d)
        p = pred.execute()
        print(p)
        cnt = 0
        frames = []
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()
    # print(len(frames))





