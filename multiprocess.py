import torch.nn as nn
from multiprocessing import Process,Queue
import os
import multiprocessing
import time
import queue
import cv2
from predict import Predict
from pytorch_i3d import InceptionI3d
import torch

i3d = InceptionI3d(400, in_channels=3)
i3d.load_state_dict(torch.load('rgb_imagenet.pt',map_location=torch.device('cpu')))
i3d.replace_logits(2000)
i3d.load_state_dict(torch.load("nslt_2000_065846_0.447803.pt",map_location=torch.device('cpu')))
# i3d.cuda()
i3d = nn.DataParallel(i3d)
i3d.eval()


def pred_func(Frames_data):
	p = Predict(Frames_data,i3d)
	p.execute()
	print(p)

def detect_realtime():
	vid = cv2.VideoCapture(0)
	cnt = 0
	original_frames = Queue()
	Frames_data = Queue()

	p1 = Process(target=pred_func,args=(Frames_data))
	p1.start()

	frames = []
	cnt = 0


	while True:
		cnt += 1
		ret,frame = vid.read()
		if not ret:
			break
		frames.append(frame)
		if cnt == 64:
			Frames_data.put(frames)
			cnt = 0

	while True:
		if Frames_data.qsize == 0:
			p1.terminate()

	cv2.destroyAllWindows()


detect_realtime()

