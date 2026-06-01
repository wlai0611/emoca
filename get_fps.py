import numpy as np
import cv2
from pathlib import Path
import argparse

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Given folder of videos and folder of NPZ files, get the FPS of each video and save it in the NPZ files")
    parser.add_argument("--video_folder", help="Path to the folder containing video files")
    parser.add_argument("--npz_folder", help="Path to the folder containing NPZ files")
    args = parser.parse_args()

    video_folder = Path(args.video_folder)
    npz_folder = Path(args.npz_folder)

    for video_path in video_folder.iterdir():
      if video_path.suffix in [".mp4", ".avi", ".mov"]:  # Check for common video file extensions
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        npz_path = npz_folder / f"{video_path.stem}.npz"
        if npz_path.exists():
          with np.load(npz_path) as data:
            data_dict = dict(data)
            data_dict['fps'] = fps
            np.savez(npz_path, **data_dict)
            print(f"Updated {npz_path} with FPS: {fps}")
        else:
            print(f"No NPZ file found for {video_path.stem}, skipping.")