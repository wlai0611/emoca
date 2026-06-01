from gdl_apps.EMOCA.utils.load import load_model
import torch
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from skimage.transform import estimate_transform, warp
import argparse
import logging
from module import make_obj
def bbox2point(left, right, top, bottom):
    old_size = (right - left + bottom - top)/2
    center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0  + old_size*0.12])
    return old_size, center

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
      if not playing:
        break
      if self.frame_counter % self.stride == 0:
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        rgb_image = cv2.resize(rgb_image, (32*(self.width//32),32*(self.height//32)))
        rgb_image = rgb_image.transpose(2,0,1)
        rgb_image = rgb_image.astype(np.float32)/255.
        yield self.frame_counter,rgb_image
    self.reader.release()

def video_to_flame_df(loader, yolo, deca, obj_folder=None, device="cpu"):
  """
  loader: torch.utils.data.DataLoader that returns frame numbers and a batch of images B*3*H*W for every iteration
  yolo: pretrained YOLO model
  deca: pretrained DECA model
  obj_folder: Path object pointing to where to save OBJ files, if omitted, we dont save OBJ files
  
  returns pandas DataFrame each row is the FLAME coefficients for a video frame
  """
  triangles = deca.deca.flame.faces_tensor #each row contains indices pointing to particular vertex of face
  resolution_inp = 224
  has_face_mask  = []
  frame_numbers  = []
  data = {'posecode':[],'cam':[],'shapecode':[],'expcode':[]} #each key stores list of tensors
  for b,(batch_idx, batch) in enumerate(loader):
    #CROP FACES
    batch_idx = batch_idx.cpu().numpy()
    batch = batch.to(device)
    frame_numbers.extend(batch_idx.tolist())
    H = batch.shape[2]
    W = batch.shape[3]
    with torch.no_grad():
      detections = yolo.predict(batch,iou=0.7,conf=0.7,verbose=False)
    processed  = torch.zeros(size=(len(batch),3,resolution_inp,resolution_inp),dtype=torch.float32,device=yolo.device.type)
    for d,detection in enumerate(detections):
      boxes = detection.boxes.xyxy
      if len(boxes) == 0:
        has_face_mask.append(0)
        best_box = torch.tensor([0,0,W,H],dtype=torch.int)
      else:
        has_face_mask.append(1)
        widths   = boxes[:,2] - boxes[:,0]
        heights  = boxes[:,3] - boxes[:,1]
        areas    = widths * heights
        best_idx = areas.argmax()
        best_box = boxes[best_idx].cpu().numpy().astype(int)
      new_size, center = get_face_center_size(best_box)
      src_pts = square_crop(center,new_size)
      DST_PTS = np.array([[0,0], [0,resolution_inp - 1], [resolution_inp - 1, 0]])
      tform   = estimate_transform('similarity', src_pts, DST_PTS)
      image   = batch[d].cpu().numpy().transpose(1,2,0)
      dst_image=warp(image, tform.inverse, output_shape=(resolution_inp, resolution_inp))
      dst_image= dst_image.transpose(2,0,1)
      processed[d] = torch.tensor(dst_image).float()
    #encode expects dict {"image":tensor[Batch*Ring*Channels*Height*Width]
    processed = {'image': processed.unsqueeze(dim=1)}#shape B*1*3*224*224
    #COMPUTE FLAME COEFFICIENTS
    with torch.no_grad():
      codedict = deca.encode(processed, training=False)
      #ADD TO CONTAINER
      for key,value in data.items():
        value.append(codedict[key])
      if obj_folder:
        with torch.no_grad():
          opdict = deca.decode(codedict, training=False)
    if obj_folder:
      obj_folder.mkdir(exist_ok=True)
      verts = opdict['verts']
      for frame_num,vert in zip(batch_idx,verts):
        make_obj(vert,triangles,obj_folder/f"{frame_num:06d}.obj")
  #concatenate list of tensors into one tensor for each key
  concatenated_data = {}
  for key, value in data.items():
    concatenated_data[key] = torch.cat(value,dim=0).cpu().numpy()
  concatenated_data['has_face'] = has_face_mask
  concatenated_data['frame_number'] = frame_numbers
  return concatenated_data

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Given a folder of videos, extract FLAME coefficients for each frame and save as .npz file")
  parser.add_argument("--videos", required=True, type=str, help="Folder of videos")
  parser.add_argument("--out", required=True, type=str, help="Folder to save FLAME coefficients as .npz file")
  parser.add_argument("--yolo", type=str, default="assets/YOLO/yolov8n-face-lindevs.pt", help="Path to YOLO weights")
  parser.add_argument("--samples_per_second", type=int, default=4, help="Number of frames to sample per second from the video")
  parser.add_argument("--batch_size", type=int, default=4, help="Batch size for processing frames")
  parser.add_argument("--emoca", type=str, default="assets/EMOCA/models", help="Path to folder containing EMOCA_v2_lr_mse_20 folder")
  parser.add_argument("--obj", action="store_true", help="Whether to save OBJ files")
  args = parser.parse_args()

  out_folder = Path(args.out)
  out_folder.mkdir(exist_ok=True)
  logging.basicConfig(filename=out_folder/"log.txt", level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

  #instantiate emoca model
  model_path = args.emoca
  emoca, conf = load_model(model_path,"EMOCA_v2_lr_mse_20","detail")
  emoca.eval()
  
  #load YOLO
  detector = YOLO(args.yolo)

  if torch.cuda.is_available():
    device = "cuda"
    emoca.cuda()
    detector.to("cuda")
  else:
    device = "cpu"
  print(device)

  video_folder = Path(args.videos)
  for video_file in video_folder.iterdir():
    if video_file.suffix not in [".mp4",".avi",".mov"]:
      continue
    #load a batch of images
    dataset = VideoIterator(video_file.as_posix(), samples_per_second=args.samples_per_second)
    loader  = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    if args.obj:
      obj = out_folder/video_file.stem
    else:
      obj = None
    data = video_to_flame_df(loader,detector,emoca,obj_folder=obj,device=device)
    filename = out_folder/f"{video_file.stem}.npz"
    np.savez(filename, frame_number = data['frame_number'], expcode = data["expcode"], 
             has_face=data["has_face"], fps = dataset.frames_per_second)
    logging.info(f"Finished processing {video_file.stem}, saved to {filename}")