import os
import cv2
import numpy as np
import trimesh
import pyrender
import argparse
from pathlib import Path

# If you're on Windows / normal desktop, you may NOT want osmesa.
# Try leaving this unset first.
# import os
# os.environ["PYOPENGL_PLATFORM"] = "egl"   # or "osmesa" on some Linux setups

def render_frame_pyrender(vertices, faces, center=None, background_black=True):
    """
    Render one frame of a VOCASET/FLAME mesh using pyrender.
    vertices: (5023, 3)
    faces:    (F, 3)
    returns:  uint8 image, shape (800, 800, 3), BGR for cv2
    """
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32)

    if center is None:
        center = vertices.mean(axis=0)

    # match their camera settings for vocaset
    camera_params = {
        "c": np.array([400, 400], dtype=np.float32),
        "f": np.array([4754.97941935 / 2, 4754.97941935 / 2], dtype=np.float32),
    }
    width, height = 800, 800

    # center the mesh roughly the same way as their script
    v = vertices.copy()
    # equivalent to rot = zeros, so no Rodrigues needed
    # just keep as-is; if you later want rotation, apply it here

    material = pyrender.MetallicRoughnessMaterial(
        alphaMode="BLEND",
        baseColorFactor=[0.3, 0.3, 0.3, 1.0],
        metallicFactor=0.8,
        roughnessFactor=0.8,
    )

    tri_mesh = trimesh.Trimesh(vertices=v, faces=faces, process=False)
    render_mesh = pyrender.Mesh.from_trimesh(tri_mesh, material=material, smooth=True)

    if background_black:
        scene = pyrender.Scene(
            ambient_light=[0.2, 0.2, 0.2],
            bg_color=[0, 0, 0]
        )
    else:
        scene = pyrender.Scene(
            ambient_light=[0.2, 0.2, 0.2],
            bg_color=[255, 255, 255]
        )

    scene.add(render_mesh, pose=np.eye(4))

    camera = pyrender.IntrinsicsCamera(
        fx=camera_params["f"][0],
        fy=camera_params["f"][1],
        cx=camera_params["c"][0],
        cy=camera_params["c"][1],
        znear=0.01,
        zfar=3.0,
    )

    # keep same-ish camera pose as original script
    camera_pose = np.eye(4, dtype=np.float32)
    camera_pose[:3, 3] = np.array([0, 0, 1.0], dtype=np.float32)
    scene.add(camera, pose=camera_pose)

    # lights
    intensity = 2.0
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=intensity)

    for light_pos in [
        [0, 0, 1.0],
        [0.2, 0, 1.0],
        [-0.2, 0, 1.0],
        [0, 0.2, 1.0],
        [0, -0.2, 1.0],
    ]:
        light_pose = np.eye(4, dtype=np.float32)
        light_pose[:3, 3] = np.array(light_pos, dtype=np.float32)
        scene.add(light, pose=light_pose)

    flags = pyrender.RenderFlags.SKIP_CULL_FACES

    try:
        renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)
        color, _ = renderer.render(scene, flags=flags)
        renderer.delete()
    except Exception as e:
        print("pyrender failed on frame:", e)
        color = np.zeros((height, width, 3), dtype=np.uint8)

    # pyrender gives RGB, cv2 wants BGR
    return color[..., ::-1]


def render_sequence_to_mp4(sequence_vertices, triangles, out_mp4, fps=30, background_black=True):
    """
    sequence_vertices: (T, 5023, 3)
    triangles: (N,3) #defines the connectivity of triangles
    """
    sequence_vertices = np.asarray(sequence_vertices)
    assert sequence_vertices.ndim == 3 and sequence_vertices.shape[2] == 3

    # load template mesh ONLY to get faces
    #template = trimesh.load(template_ply, process=False)
    #faces = np.asarray(template.faces)

    T = sequence_vertices.shape[0]
    writer = cv2.VideoWriter(
        out_mp4,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (800, 800),
        True
    )

    center = sequence_vertices[0].mean(axis=0)

    for i in range(T):
        frame = render_frame_pyrender(
            sequence_vertices[i],
            triangles,
            center=center,
            background_black=background_black
        )
        writer.write(frame)
        if i % 20 == 0:
            print(f"rendered {i}/{T}")

    writer.release()
    print("saved:", out_mp4)

def mux_audio_video(video_path, wav_path, out_path):
    """
    Combine silent MP4 + WAV into final MP4 using ffmpeg.
    """
    import subprocess
    cmd = [
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-i", wav_path,
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        out_path
    ]
    subprocess.run(cmd, check=True)

if __name__=="__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--npz",help='path to NPY file of predictions from FaceFormer')
  #parser.add_argument("--wav",help="WAV file of audio")
  args = parser.parse_args()
  npy_file = Path(args.npz)
  parent_folder = npy_file.parent
  silent_mp4_file = parent_folder/f"{npy_file.stem}_render.mp4"
  audio_mp4_file  = parent_folder/f"{npy_file.stem}_audio.mp4"
  pred = np.load(npy_file)
  verts = pred['verts']
  triangles = pred['triangles']
  # if flattened, reshape it
  #if pred.ndim == 2 and pred.shape[1] == 5023 * 3:
  #  pred = pred.reshape(-1, 5023, 3)

  render_sequence_to_mp4(
    sequence_vertices=verts,
    triangles=triangles,
    out_mp4=silent_mp4_file,
    fps=4,   # FaceFormer used 30 for vocaset rendering script
    background_black=True
  )
  #mux_audio_video(silent_mp4_file.as_posix(),args.wav,audio_mp4_file.as_posix())