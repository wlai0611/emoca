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
import argparse
import torch
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

def get_tracks(yolo, video, sample_frequency=4, outvideo=None, batch_size=4):
  '''
  Video is the Path object pointing to MP4 or AVI video
  sample_frequency = how many bounding boxes to get per second
  outvideo is the path to create another video in which bounding boxes overlayed on faces, default is omitted
  Returns
  tracks: dictionary tracking the location and timing of each persistent face in video:
  { 1:{bbox: [bbox0,bbox1,...,bboxT], frame_nums:[t0,t1,...tT],}, 
    2:{bbox: [bbox0,bbox1,...,bboxT], frame_nums:[t0,t1,...tT],}
  }
  images: dictionary mapping frame_num to actual image
  { 123: np.array, 234: np.array} 
  metadata: dict of height, width and number of frames of image, sample_frequency
  '''
  tracks = {}
  
  if 'bbox' in video.name:
    return None, None, None
  dataset= VideoIterator(video_file=video.as_posix(), samples_per_second=sample_frequency)
  loader  = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
  
  fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

  blue   = (33,29,159)#
  green  = (0,128,255) 
  verde  = (141,141,35)
  white  = (255,255,255)
  colors = {0:blue,1:green,2:verde,3:white}
  font  = cv2.FONT_HERSHEY_SIMPLEX
  images = {}
  for b,(frames, times, processed_imgs, raw_imgs) in enumerate(loader):
    _,height,width,channels = raw_imgs.shape
    processed_imgs = processed_imgs.to(device)
    if b==0 and outvideo:
      writer = cv2.VideoWriter(outvideo.as_posix(), fourcc, dataset.samples_per_second, (width,height))#704, 384
    result_per_img = yolo.track(processed_imgs,persist=True,iou=0.7,conf=0.8,verbose=False)
    for frame_num,t,img_boxes,raw_img,processed_img in zip(frames,times,result_per_img,raw_imgs,processed_imgs):
      images[frame_num.item()] = processed_img.cpu()
      channels,processed_height,processed_width=processed_img.shape
      faces = img_boxes.boxes
      if faces.id is None:
        continue
      track_ids = faces.id.cpu().numpy().astype(int).tolist()
      raw_img = raw_img.cpu().numpy().astype(np.uint8)
      for track_id, face in zip(track_ids,faces):
        if track_id not in tracks:
          tracks[track_id] = {'t':[],'bbox':[],'frame_nums':[],'cumulative_area':0}
        xyxy=face.xyxy.cpu().numpy().astype(int).tolist()
        x1,y1,x2,y2=xyxy[0]
        area = (x2-x1)*(y2-y1)
        if x2<=x1 or y2<=y1:
          continue
        if x2<=0 or y2<=0 or x1>=processed_width or y1>=processed_height:
          continue
        tracks[track_id]['frame_nums'].append(frame_num.item())
        tracks[track_id]['t'].append(t.item())
        tracks[track_id]['bbox'].append(xyxy)
        tracks[track_id]['cumulative_area'] += area      
        color=colors.get(track_id,(0,0,0))
        if outvideo:
          cv2.rectangle(raw_img,(x1,y1),(x2,y2),color=color)   
          cv2.putText(raw_img, str(track_id),(x2,y2),font,1,color,2,cv2.LINE_AA) 
      if outvideo:
        writer.write(raw_img) 
  if outvideo:
    writer.release()

  channels, height, width = processed_img.shape
  metadata = {}
  metadata['height'] = height
  metadata['width']  = width
  metadata['total_frames'] = dataset.total_frames
  return tracks, images, metadata

def get_blendshapes(track, images, batch_size):
  #track has format {bbox: [bbox0,bbox1,...,bboxT], frame_nums:[t0,t1,...tT],}
  #images has format { 123: np.array, 234: np.array} each array is image 3*H*W
  crop_dataset = BboxIterator(track, images)
  crop_loader  = torch.utils.data.DataLoader(crop_dataset, batch_size=batch_size, shuffle=False)

  timestamps = []
  vert_series = []
  posecode  = []
  shapecode = []
  expcode   = []
  for timesteps,crops in crop_loader:
    timestamps.extend(timesteps.tolist())
    processed = {'image': crops.unsqueeze(dim=1).to(device)}
    with torch.no_grad():
      codedict = emoca.encode(processed, training=False)
    posecode.append(codedict['posecode'].clone().detach().cpu())
    shapecode.append(codedict['shapecode'].clone().detach().cpu())
    expcode.append(codedict['expcode'].clone().detach().cpu())
    codedict['shapecode'][:,:] = 0
    codedict['posecode'][:,:3] = 0
    with torch.no_grad():
      opdict = emoca.decode(codedict, training=False)
    verts = opdict['verts'].detach().cpu()
    vert_series.append(verts)
  shapecode = torch.concatenate(shapecode).numpy()
  posecode = torch.concatenate(posecode).numpy()
  expcode = torch.concatenate(expcode).numpy()
  vert_series = torch.concatenate(vert_series).numpy()
  return timestamps,vert_series,posecode,shapecode,expcode

def get_best_track(tracks,metadata,coverage_weight = 1.,area_weight = 1.):
  '''
    rank the face-tracks based on 
    1) how much of the screen face occupies (area)
    2) how long the face stays on screen (coverage)
    return best track
  '''
  height = metadata['height']
  width  = metadata['width']
  total_area = height*width
  nframe = metadata['total_frames']
  top_2_tracks = []
  for track_id,track in tracks.items():
    track['score']=(area_weight*track["cumulative_area"]/total_area)+(coverage_weight*len(track["t"])/nframe)
    top_2_tracks.append((track_id,track['score']))
    top_2_tracks = sorted(top_2_tracks,key = lambda tupl:tupl[1],reverse=True)[:2]
  #decide if video should be kept
  best_track = tracks[top_2_tracks[0][0]]
  runner_up  = tracks[top_2_tracks[1][0]]
  if (best_track['score']-runner_up['score']) < 0.5:
    return None
  return best_track

def save_mid_frame_with_bbox(track, video):
  middle_frame = track['frame_nums'][len(track['frame_nums'])//2]
  middle_bbox  = track['bbox'][len(track['frame_nums'])//2]
  (x1,y1,x2,y2), = middle_bbox
  middle_image = get_frame(video.as_posix(), middle_frame)
  cv2.rectangle(middle_image,(x1,y1),(x2,y2),color=(0,0,255))
  return middle_image

parser=argparse.ArgumentParser(description="given a folder of videos, run face tracker, then EMOCA on the cropped face")
parser.add_argument("--infolder",help="path to folder containing videos")
parser.add_argument("--outfolder",help="path to folder to save processed data")
parser.add_argument("--freq",help='number of blendshapes to create per second of video',type=int,default=4)
parser.add_argument("--boxvids",action='store_true',help='whether save videos with bboxes')
parser.add_argument("--batch_size",type=int,default=4,help="How many images to EMOCA at once")
args = parser.parse_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
vidfolder= Path(args.infolder)
outfolder= Path(args.outfolder)
outfolder.mkdir(exist_ok=True)
logging.basicConfig(filename=outfolder/f"preprocessing{time.time()}.log", level = logging.INFO)
model_path = "assets/EMOCA/models"
emoca, conf = load_model(model_path,"EMOCA_v2_lr_mse_20","detail")
emoca.eval()
emoca.to(device)
triangles = emoca.deca.flame.faces_tensor.cpu().numpy()

videos = list(vidfolder.glob("*.mp4"))+list(vidfolder.glob("*.avi"))

logging.info(f"device {device}")
logging.info(f"{len(videos)} videos found")
for vidnum,video in enumerate(videos):
  try:  
    subfolder = outfolder/video.stem
    subfolder.mkdir(exist_ok=True)
    start = time.time()
    if args.boxvids:
      outvideo = subfolder/"tracking.mp4"
    else:
      outvideo = None
    yolo   = YOLO('assets/YOLO/yolov8n-face-lindevs.pt')
    yolo.to(device)
    tracks,images,metadata = get_tracks(yolo,video,sample_frequency=args.freq,outvideo=outvideo,batch_size=args.batch_size)
    if not tracks:
      logging.info(f'No face found in {video.as_posix()}')
      continue
    if len(tracks) == 1:
      best_track = list(tracks.values()).pop()
    else:
      best_track = get_best_track(tracks, metadata)
      if best_track is None:
        logging.info(f"{video.stem} too many faces")
        continue
    middle_image = save_mid_frame_with_bbox(best_track,video)
    cv2.imwrite(subfolder/f"img_for_prompt.jpg",middle_image)
    print("###")
    print(video.name)
    print("###")

    timestamps, vert_series,posecode,shapecode,expcode = get_blendshapes(best_track, images, batch_size=args.batch_size)
    np.savez(subfolder/"blendshapes.npz",triangles=triangles, verts=vert_series, times=timestamps, freq=args.freq,posecode=posecode,shapecode=shapecode,expcode=expcode)
    json.dump(tracks,open(subfolder/"face_tracks.json","w"))
    logging.info(f"{video.stem} finished in {time.time()-start} seconds")
  except Exception as e:
    logging.exception(f"{video.stem} error with {e.args} and {e.__class__.__name__} on line {e.__traceback__.tb_lineno}")

""" writer = cv2.VideoWriter(outvideo.as_posix(), fourcc, dataset.samples_per_second, (704, 384))
for i,(time, frame, raw) in enumerate(loader):
  for img in raw:
    writer.write(img.numpy().astype(np.uint8))
writer.release() """