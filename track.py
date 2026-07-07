from ultralytics import YOLO
from pathlib import Path
import cv2
import torch
import numpy as np
from skimage.transform import estimate_transform, warp

class VideoIterator(torch.utils.data.IterableDataset):
  def __init__(self,video_file,samples_per_second=1):
    self.video_file = video_file
    self.samples_per_second = samples_per_second
    self.frame_counter = 0
  def __iter__(self):
    self.reader = cv2.VideoCapture(self.video_file)
    self.height = int(self.reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    self.width  = int(self.reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    self.frames_per_second = int(self.reader.get(cv2.CAP_PROP_FPS))
    self.stride = max(1,self.frames_per_second//self.samples_per_second)
    while True:
      playing, bgr_image = self.reader.read()
      self.frame_counter += 1
      self.timer = self.frame_counter/self.frames_per_second
      if not playing:
        break
      bgr_image = cv2.resize(bgr_image, (32*(self.width//32),32*(self.height//32)))
      if self.frame_counter % self.stride == 0:
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        rgb_image = rgb_image.transpose(2,0,1)
        rgb_image = rgb_image.astype(np.float32)/255.
        yield self.frame_counter,self.timer,rgb_image, bgr_image
    self.reader.release()

def get_face_center_size(box,scale=1.25):
  l,t,r,b = box
  old_size= (r-l+b-t)/2
  center  = (l+r)/2. , (t+b)/2. + old_size*0.12 #shift down 12%
  return scale*old_size, center  

def square_crop(center, size):
  corner1 = center[0]-size/2,center[1]-size/2
  corner2 = center[0]-size/2,center[1]+size/2
  corner3 = center[0]+size/2,center[1]-size/2
  return np.array([corner1,corner2,corner3])

def process_face_crops(crops,resolution_inp=224):
    #crops is Batch*Channels*Height*Width
    processed  = torch.zeros(size=(len(crops),3,resolution_inp,resolution_inp),dtype=torch.float32,device=yolo.device.type)
    for d,crop in enumerate(crops):
      channels,height,width = crop.shape
      best_box = (0,0,width,height)
      new_size, center = get_face_center_size(best_box)
      src_pts = square_crop(center,new_size)
      DST_PTS = np.array([[0,0], [0,resolution_inp - 1], [resolution_inp - 1, 0]])
      tform   = estimate_transform('similarity', src_pts, DST_PTS)
      image   = crop.cpu().numpy().transpose(1,2,0)
      dst_image=warp(image, tform.inverse, output_shape=(resolution_inp, resolution_inp))
      dst_image= dst_image.transpose(2,0,1)
      processed[d] = torch.tensor(dst_image).float()
    return processed

model  = YOLO('assets/YOLO/yolov8n-face-lindevs.pt')
video  = Path("../clooney/0489.avi")
dataset= VideoIterator(video_file=video.as_posix(), samples_per_second=4)
loader  = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False)

tracks = {} #{Track1:[bbox0,bbox1], Track2:[bbox0,bbox1]}
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
fps = 4
outvideo = Path("../clooney/test.mp4")

blue   = (33,29,159)
green  = (0,128,255) 
verde  = (141,141,35)
white  = (255,255,255)
colors = {0:blue,1:green,2:verde,3:white}

""" writer = cv2.VideoWriter(outvideo.as_posix(), fourcc, dataset.samples_per_second, (704, 384))
for i,(time, frame, raw) in enumerate(loader):
  for img in raw:
    writer.write(img.numpy().astype(np.uint8))
writer.release() """
images = {}
for b,(frames, times, processed_imgs, raw_imgs) in enumerate(loader):
  _,height,width,channels = raw_imgs.shape
  if b==0:
    writer = cv2.VideoWriter(outvideo.as_posix(), fourcc, dataset.samples_per_second, (width,height))#704, 384
  result_per_img = model.track(processed_imgs,persist=True,iou=0.7,conf=0.7,verbose=False)
  for frame_num,t,img_boxes,raw_img,processed_img in zip(frames,times,result_per_img,raw_imgs,processed_imgs):
    images[frame_num.item()] = processed_img
    faces = img_boxes.boxes
    if faces.id is None:
      continue
    track_ids = faces.id.numpy().astype(int)
    raw_img = raw_img.numpy().astype(np.uint8)
    for track_id, face in zip(track_ids,faces):
      if track_id not in tracks:
        tracks[track_id] = {'t':[],'bbox':[],'frame_nums':[],'cumulative_area':0}
      xyxy=face.xyxy.numpy().astype(int)
      x1,y1,x2,y2=xyxy[0]
      area = (x2-x1)*(y2-y1)
      tracks[track_id]['frame_nums'].append(frame_num.item())
      tracks[track_id]['t'].append(t.item())
      tracks[track_id]['bbox'].append(xyxy)
      tracks[track_id]['cumulative_area'] += area      
      cv2.rectangle(raw_img,xyxy[0,:2],xyxy[0,2:],color=colors.get(track_id,(0,0,0)))    
    writer.write(raw_img) 
writer.release()
best_track_id,best_track = max(list(tracks.items()), key = lambda tupl:tupl[1]['cumulative_area'])
for frame_num,timestep, bbox in zip(best_track["frame_nums"],best_track["t"],best_track["bbox"]):
  pass
print()
#tracks[track_id].append({'t':})  
# #add up areas right after detedction 
#if multiple tracks have area-times > threshold, discard clip 
print()