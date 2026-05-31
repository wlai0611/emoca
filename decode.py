from module import make_obj
import argparse
from pathlib import Path
import numpy as np
import torch
from gdl_apps.EMOCA.utils.load import load_model

class NPZIterator(torch.utils.data.IterableDataset):
  def __init__(self,npz_file):
    self.npz_file = npz_file
  def __iter__(self):
    data = np.load(self.npz_file)
    frame_numbers = data['frame_number']
    codedict = {"expcode": torch.from_numpy(data["expcode"]),
                "shapecode": torch.zeros((data["expcode"].shape[0],100)),
                "posecode": torch.zeros((data["expcode"].shape[0],6)),
                "cam": torch.zeros((data["expcode"].shape[0],3)),
                'detailcode':torch.zeros((data["expcode"].shape[0],128)),
                'detailemocode':torch.zeros((data["expcode"].shape[0],0)),
                'images':torch.zeros((data["expcode"].shape[0],3,224,224)),
                'texcode':torch.zeros((data["expcode"].shape[0],50)),
                'lightcode':torch.zeros((data["expcode"].shape[0],9,3)),
    }     
    for i,frame_number in enumerate(frame_numbers):
      yield frame_number, {k:v[i] for k,v in codedict.items()}

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Given a folder of NPZ files containing FLAME coefficients, decode them into 3D meshes and save as .obj files") 
  parser.add_argument("--npz", required=True, type=str, help="Folder containing .npz files with FLAME coefficients")
  parser.add_argument("--obj", required=True, type=str, help="Folder to save subfolders of OBJ files for each .npz file")
  parser.add_argument("--emoca", type=str, default="assets/EMOCA/models", help="Path to folder containing EMOCA_v2_lr_mse_20 folder")
  parser.add_argument("--batchsize", type=int, default=4, help="Batch size for decoding FLAME coefficients")
  args = parser.parse_args()
  npz_folder = Path(args.npz)
  obj_folder = Path(args.obj)
  obj_folder.mkdir(exist_ok=True)

  #instantiate emoca model
  model_path = args.emoca
  emoca, conf = load_model(model_path,"EMOCA_v2_lr_mse_20","detail")
  emoca.eval()  
  triangles = emoca.deca.flame.faces_tensor.cpu().numpy()

  for npz_file in npz_folder.iterdir():
    if npz_file.suffix != ".npz":
      continue
    subfolder = obj_folder/npz_file.stem
    subfolder.mkdir(exist_ok=True)
    dataset = NPZIterator(npz_file.as_posix())
    loader  = torch.utils.data.DataLoader(dataset, batch_size=args.batchsize, shuffle=False)
    for frame_numbers, codedict in loader:
      with torch.no_grad():
        opdict = emoca.decode(codedict, training=False)  
        verts = opdict['verts']
        for frame_num,vert in zip(frame_numbers,verts):
          make_obj(vert,triangles,subfolder/f"{frame_num:06d}.obj")    