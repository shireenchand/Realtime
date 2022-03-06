import cv2
from predict import Predict
from pytorch_i3d import InceptionI3d
import torch
import torch.nn as nn
from multiprocessing import Pool
import os
import multiprocessing
import time
import threading
import json
import requests
# import language_tool_python


# tool = language_tool_python.LanguageTool('en-US')
def correct_sent_tc_api(sent):
  url = "https://rewriter-paraphraser-text-changer-multi-language.p.rapidapi.com/rewrite"
  
  payload = "{\r\"language\": \"en\",\r\"strength\": 3,\r\"text\": \""+sent+"\"\r}"
  headers = {
      'content-type': "application/json",
      'x-rapidapi-host': "rewriter-paraphraser-text-changer-multi-language.p.rapidapi.com",
      'x-rapidapi-key': "ac09d369admsh68d8a0d46565964p12a32cjsn1e6a2f83ac0b"
      }
  response = requests.request("POST", url, data=payload, headers=headers)
  return response.text

coordinates = (100,100)
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
color = (255,0,255)
thickness = 2
global res
res = "Detecting....."

global previous
previous = []

global p
global words
words = []
p = "hello"

def send(frames,i3d):
  pred = Predict(frames,i3d)
  p = pred.execute()
  print(p)
  words.append(str(p))

with open('wlasl_class_list.txt') as f:
  labels = f.readlines()
map = {}
for element in labels:
  first = element.split('\t')
  second = first[1].split('\n')[0]
  map[first[0]] = second

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
x = threading.Thread()
while(True):
    ret,frame = vid.read()
    cnt += 1
    cv2.putText(frame, "Press e to finish a sentence", (100,20), font, fontScale, color, thickness, cv2.LINE_AA)
    cv2.putText(frame, res, coordinates, font, fontScale, color, thickness, cv2.LINE_AA)
    cv2.imshow("frame",frame)
    # cv2.putText(frame, "hello", (10,500), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4, cv2.LINE_AA)
    frames.append(frame)
    if cnt == 64:
      if not x.is_alive():
        x = threading.Thread(target=send,args=(frames,i3d))
        x.start()
      else:
        pass
        # print("64 Frames")
        # pred = Predict(frames,i3d)
        # p = pred.execute()
      # print(p)
      cnt = 0
      frames = []
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if cv2.waitKey(1) & 0xFF == ord("e"):
      print("end")
      res = "Predicting......."
      # if words == []:
      #   text_words = previous
      # else:
      #   text_words = [map[num] for num in words]
      # words = []
      # print(text_words)
      print(words)
      text_words = [map[num] for num in words]
      words = []
      # text_words = previous
      s = " ".join(text_words)
      print(s)
      res=correct_sent_tc_api(s)
      res_dict=json.loads(res)
      res = res_dict["rewrite"]
      print("Details :",res)

      # print("Final :",tool.correct(res_dict['rewrite']))

vid.release()
cv2.destroyAllWindows()
    # print(len(frames))


