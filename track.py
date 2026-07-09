from ultralytics import YOLO
from pathlib import Path
import cv2
import torch
import numpy as np
from skimage.transform import estimate_transform, warp
from gdl_apps.EMOCA.utils.load import load_model
import logging
import time
import json
def get_frame(video_file, frame_idx):
  reader = cv2.VideoCapture(video_file)
  num_frames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
  if num_frames <= 0:
    reader.release()
    return None
  reader.set(cv2.CAP_PROP_POS_FRAMES,frame_idx)
  success,frame = reader.read()
  reader.release()
  if success:
    return frame
  else:
    logging.info(f"{video_file} failed frame seeking. Looping through video for frame.")
    reader = cv2.VideoCapture(video_file)
    frame_counter = 0
    while True:
      success,frame = reader.read()
      frame_counter += 1
      if success:
        if frame_counter == frame_idx:
          break
      else:
        frame = None
        break
    reader.release()
    return frame
    

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
    self.total_frames = int(self.frame_counter/self.stride)

class BboxIterator(torch.utils.data.Dataset):
  def __init__(self,track_dict, frames, resolution_inp = 224):
    self.track_dict=track_dict
    self.frames = frames
    self.resolution_inp = resolution_inp
  def __len__(self):
    return len(self.track_dict['t'])
  
  def get_face_center_size(self,box,scale=1.25):
    l,t,r,b = box
    old_size= (r-l+b-t)/2
    center  = (l+r)/2. , (t+b)/2. + old_size*0.12 #shift down 12%
    return scale*old_size, center  

  def square_crop(self,center, size):
    corner1 = center[0]-size/2,center[1]-size/2
    corner2 = center[0]-size/2,center[1]+size/2
    corner3 = center[0]+size/2,center[1]-size/2
    return np.array([corner1,corner2,corner3])
  
  def __getitem__(self, index):
    frame_num = self.track_dict['frame_nums'][index]
    frame = self.frames[frame_num]
    timestep = self.track_dict['t'][index]
    x1,y1,x2,y2 = self.track_dict['bbox'][index][0]
    crop = frame[:,y1:y2,x1:x2]
    channels,height,width = crop.shape
    best_box = (0,0,width,height)
    new_size, center = self.get_face_center_size(best_box)
    src_pts = self.square_crop(center,new_size)
    DST_PTS = np.array([[0,0], [0,self.resolution_inp - 1], [self.resolution_inp - 1, 0]])
    tform   = estimate_transform('similarity', src_pts, DST_PTS)
    image   = crop.cpu().numpy().transpose(1,2,0)
    dst_image=warp(image, tform.inverse, output_shape=(self.resolution_inp, self.resolution_inp))
    dst_image= dst_image.transpose(2,0,1)
    return timestep, dst_image

logging.basicConfig(filename=f"preprocessing{time.time()}.log", level = logging.INFO)
vidfolder= Path("../test_videos")
font  = cv2.FONT_HERSHEY_SIMPLEX
coverage_weight = 1.
area_weight = 1.

model_path = "assets/EMOCA/models"
emoca, conf = load_model(model_path,"EMOCA_v2_lr_mse_20","detail")
emoca.eval()

videos = list(vidfolder.glob("*.mp4"))+list(vidfolder.glob("*.avi"))
npz_folder = vidfolder/"npz"
tracking_folder = vidfolder/"tracking"
npz_folder.mkdir(exist_ok=True)
tracking_folder.mkdir(exist_ok=True)
for video in videos:
  start = time.time()
  model  = YOLO('assets/YOLO/yolov8n-face-lindevs.pt')
  if 'bbox' in video.name:
    continue
  #video  = Path("../clooney/0489.avi")
  dataset= VideoIterator(video_file=video.as_posix(), samples_per_second=4)
  loader  = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False)
  tracks = {} #{Track1:[bbox0,bbox1], Track2:[bbox0,bbox1]}
  fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
  fps = 4
  outvideo = vidfolder/f"{video.stem}_bbox.mp4"

  blue   = (33,29,159)
  green  = (0,128,255) 
  verde  = (141,141,35)
  white  = (255,255,255)
  colors = {0:blue,1:green,2:verde,3:white}

  images = {}
  for b,(frames, times, processed_imgs, raw_imgs) in enumerate(loader):
    _,height,width,channels = raw_imgs.shape
    if b==0:
      writer = cv2.VideoWriter(outvideo.as_posix(), fourcc, dataset.samples_per_second, (width,height))#704, 384
    result_per_img = model.track(processed_imgs,persist=True,iou=0.7,conf=0.8,verbose=False)
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
        color=colors.get(track_id,(0,0,0))
        cv2.rectangle(raw_img,xyxy[0,:2],xyxy[0,2:],color=color)   
        cv2.putText(raw_img, str(track_id),xyxy[0,[2,3]],font,1,color,2,cv2.LINE_AA) 
      writer.write(raw_img) 
  writer.release()

  if not tracks:
    logging.info(f'No face found in {video.as_posix()}')
    continue

  channels, height, width = processed_img.shape
  total_area = height*width
  nframe = dataset.total_frames

  if len(tracks) == 1:
    best_track = list(tracks.values()).pop()
  
  else:
    top_2_tracks = []
    for track_id,track in tracks.items():
      track['score']=(area_weight*track["cumulative_area"]/total_area)+(coverage_weight*len(track["t"])/nframe)
      top_2_tracks.append((track_id,track['score']))
      top_2_tracks = sorted(top_2_tracks,key = lambda tupl:tupl[1],reverse=True)[:2]
    #decide if video should be kept
    best_track = tracks[top_2_tracks[0][0]]
    runner_up  = tracks[top_2_tracks[1][0]]
    if (best_track['score']-runner_up['score']) < 0.5:
      logging.info(f"{video.stem} too many faces.  Scores {best_track['score']} and {runner_up['score']}")
      continue
  middle_frame = best_track['frame_nums'][len(best_track['frame_nums'])//2]
  middle_bbox  = best_track['bbox'][len(best_track['frame_nums'])//2]
  middle_image = get_frame(video.as_posix(), middle_frame)
  cv2.rectangle(middle_image,middle_bbox[0,:2],middle_bbox[0,2:],color=(0,0,255))
  cv2.imwrite(video.parent/f"{video.stem}.jpg",middle_image)
  print("###")
  print(video.name)
  print("###")

  crop_dataset = BboxIterator(best_track, images)
  crop_loader  = torch.utils.data.DataLoader(crop_dataset, batch_size=4, shuffle=False)

  times = []
  vert_series = []
  for timesteps,crops in crop_loader:
    times.extend(timesteps.tolist())
    processed = {'image': crops.unsqueeze(dim=1)}
    with torch.no_grad():
      codedict = emoca.encode(processed, training=False)
    with torch.no_grad():
      opdict = emoca.decode(codedict, training=False)
    verts = opdict['verts']
    vert_series.append(verts)
  vert_series = torch.concatenate(vert_series).numpy()
  triangles = emoca.deca.flame.faces_tensor.numpy() 
  np.savez(npz_folder/f"{video.stem}.npz",triangles=triangles, verts=vert_series, times=times)
  json.dump(tracks,open(tracking_folder/f"{video.stem}.json","w"))
  logging.info(f"{video.stem} finished in {time.time()-start} seconds")

""" writer = cv2.VideoWriter(outvideo.as_posix(), fourcc, dataset.samples_per_second, (704, 384))
for i,(time, frame, raw) in enumerate(loader):
  for img in raw:
    writer.write(img.numpy().astype(np.uint8))
writer.release() """