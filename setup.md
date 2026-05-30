This is the set up that ChatGPT created to 
run EMOCA *only on inference* on Windows without any training capabilities (which require a lot more setup)  

Create a Python 3.9 venv (I am using Git Bash on Windows to run this)
```
py -3.9 -m venv emoca_env
source emoca_env/Scripts/activate
```  
Install bare minimum dependencies  
```
pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1
pip install omegaconf==2.0.6
pip install chumpy==0.70
pip install numpy==1.23.5
pip install torchmetrics==0.6.2
pip install pytorch-lightning==1.4.9
pip install opencv-python==4.5.5.64
```

Now every time I want to run in inference, to avoid 
using the `pip install -e .` which I fear will install more things, I just set the path forcefully  
`export PYTHONPATH=$PYTHONPATH:$(pwd)`  

and then I run the demo 
`python -m gdl_apps.EMOCA.demos.test_emoca_on_images`
which expects the following files (assuming `emoca` is root)  
  the list of files was inferred from [this PY file](gdl_apps/EMOCA/utils/load.py) see their `replace_asset_dirs` function
```
/assets/data/EMOCA_test_example_data/images/affectnet_test_examples/(images go here)  
/assets/DECA/data/deca_model.tar  
/assets/EMOCA/models/EMOCA_v2_lr_mse_20
/assets/FaceRecognition/resnet50_ft_weight.pkl  
/assets/FLAME/geometry/fixed_uv_displacements/fixed_displacement_256.npy  
/assets/FLAME/geometry/generic_model.pkl  
/assets/FLAME/geometry/head_template.obj  
/assets/FLAME/geometry/landmark_embedding.npy  
/assets/FLAME/geometry/mediapipe_landmark_embedding.npz  
/assets/FLAME/mask/uv_face_eye_mask.png  
/assets/FLAME/mask/uv_face_mask.png  
```
which are downloaded from (if blocks u must create accoutn at [](https://flame.is.tue.mpg.de/)) see their [file](gdl_apps/EMOCA/demos/download_assets.sh)  
1. [EMOCA_v2_lr_mse_20](https://download.is.tue.mpg.de/emoca/assets/EMOCA/models/EMOCA_v2_lr_mse_20.zip)  
2. [deca_model.tar](https://download.is.tue.mpg.de/emoca/assets/EMOCA/models/DECA.zip)  
3. [DECA](https://download.is.tue.mpg.de/emoca/assets/DECA.zip)  
4. [FaceRecognition](https://download.is.tue.mpg.de/emoca/assets/FaceRecognition.zip)  
5. [FLAME](https://download.is.tue.mpg.de/emoca/assets/FLAME.zip)  
6. [EMOCA_test_example_data](https://download.is.tue.mpg.de/emoca/assets/data/EMOCA_test_example_data.zip)  