from pytorch_i3d import InceptionI3d
import torch.nn as nn
import torch
import os
import math
import random
from torchvision import transforms
import videotransforms
import cv2
import numpy as np


class Predict():
	def __init__(self,frames_given,i3d):
		self.frames_given = frames_given
		self.i3d = i3d
		# self.i3d = InceptionI3d(400, in_channels=3)
		# self.i3d.load_state_dict(torch.load('rgb_imagenet.pt',map_location=torch.device('cpu')))
		# self.i3d.replace_logits(2000)
		# self.i3d.load_state_dict(torch.load("nslt_2000_065846_0.447803.pt",map_location=torch.device('cpu')))
		# # i3d.cuda()
		# self.i3d = nn.DataParallel(self.i3d)
		# self.i3d.eval()


	def load_rgb_frames_from_video(self,frames_given,start, num, resize=(256, 256)):
	    # video_path = os.path.join(vid_root, vid + '.mp4')
	    # print(video_path)
	    # vidcap = cv2.VideoCapture(video_path)
	    total_frames = 64
	    frames = []

	    # total_frames = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
	    cnt = 0
	    # vidcap.set(cv2.CAP_PROP_POS_FRAMES, start)
	    for offset in range(min(num, int(total_frames - start))):
	        # success, img = vidcap.read()
	        img = frames_given[cnt]

	        ## Add by Ray - get out if we have no more frames
	        # if success == False:
	        #     break

	        w, h, c = img.shape
	        if w < 226 or h < 226:
	            d = 226. - min(w, h)
	            sc = 1 + d / min(w, h)
	            img = cv2.resize(img, dsize=(0, 0), fx=sc, fy=sc)

	        if w > 256 or h > 256:
	            img = cv2.resize(img, (math.ceil(w * (256 / w)), math.ceil(h * (256 / h))))

	        img = (img / 255.) * 2 - 1

	        frames.append(img)
	        cnt += 1

	    return np.asarray(frames, dtype=np.float32)

	def video_to_tensor(self,pic):
	    """Convert a ``numpy.ndarray`` to tensor.
	    Converts a numpy.ndarray (T x H x W x C)
	    to a torch.FloatTensor of shape (C x T x H x W)
	    
	    Args:
	         pic (numpy.ndarray): Video to be converted to tensor.
	    Returns:
	         Tensor: Converted video.
	    """
	    return torch.from_numpy(pic.transpose([3, 0, 1, 2]))


	def pad(self,imgs, total_frames):
	  if imgs.shape[0] < total_frames:
	    num_padding = total_frames - imgs.shape[0]

	    if num_padding:
	      prob = np.random.random_sample()
	      if prob > 0.5:
	        pad_img = imgs[0]
	        pad = np.tile(np.expand_dims(pad_img, axis=0), (num_padding, 1, 1, 1))
	        global padded_imgs
	        padded_imgs = np.concatenate([imgs, pad], axis=0)
	      else:
	        pad_img = imgs[-1]
	        pad = np.tile(np.expand_dims(pad_img, axis=0), (num_padding, 1, 1, 1))
	        padded_imgs = np.concatenate([imgs, pad], axis=0)
	  else:
	    padded_imgs = imgs

	  # print(padded_imgs) 
	  return padded_imgs

	def execute(self):
		start_frame = 1
		total_frames = 64
		nf = 173
		root = {'word':'/content/videos'}
		start_f = random.randint(0, nf - total_frames - 1) + start_frame
		imgs = self.load_rgb_frames_from_video(frames_given=self.frames_given, start=start_f, num=total_frames)
		test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])
		total_frames = 64
		if imgs.shape[0] == 0:
		  imgs = np.zeros((total_frames, 224, 224, 3), dtype=np.float32)
		  print(os.path.join(root['word'], "vid" + '.mp4') + ' could not be read for some reason.  Skipping')
		  label = -1
		else:
		            # If we don't end up having 64 frames, then we pad the video sequence
		            # to get up to 64
		            # We randomly choose the first or last frame and tack it onto the end
		  imgs = self.pad(imgs, total_frames)
		            
		            # Run through the data augmentation
		            # 64 x 224 x 224 x 3
		  imgs = test_transforms(imgs)


		ret_img = self.video_to_tensor(imgs)

		ret_img = torch.reshape(ret_img, (1, 3, 64, 224, 224))

		per_frame_logits = self.i3d(ret_img)
		predictions = torch.max(per_frame_logits, dim=2)[0]
		out_labels = np.argsort(predictions.cpu().detach().numpy()[0])
		# print(out_labels[-1])
		return(out_labels[-1])

