import pickle
import os
import h5py
import numpy as np
import open3d as o3d
try:
    from snapshot_smpl.renderer import Renderer
except ModuleNotFoundError:
    from tools.snapshot_smpl.renderer import Renderer
import cv2
import tqdm
from pathlib import Path

def render_smpl(vertices, img, K, R, T):
    rendered_img = renderer.render_multiview(vertices, K[None], R[None],
                                             T[None, None], [img])[0]
    return rendered_img

data_root = Path('data/zju_mocap/')
subject_id = 377
ratio = 0.5
distort = False

subject_path = data_root / f"CoreView_{subject_id}"
annotation_path = subject_path / "annots.npy"
mask_dir = subject_path / "mask_cihp"
vertices_dir = subject_path / "new_vertices"


annots = np.load(annotation_path, allow_pickle=True).item()
cams = annots['cams']
ims = annots['ims']

num_cams = len(cams['K'])
num_frames = len(ims)

cam_id = 0

K = cams['K'][cam_id] 
D = cams['D'][cam_id]
D = D.reshape(-1)
T = cams['T'][cam_id]
T = T / 1000.
T = T.reshape(-1)
R = cams['R'][cam_id]
K[:2] = K[:2] * ratio
import pdb;pdb.set_trace()

smpl_path = data_root / "SMPL_NEUTRAL.pkl"
with open(smpl_path, 'rb') as smpl_file:
    _data = pickle.load(smpl_file, encoding='latin1')
faces = _data['f']

renderer = Renderer(height=int(1024 * ratio + 0.5), width=int(1024 * ratio + 0.5), faces=faces)
out_dir = Path("tmp")
out_dir.mkdir(exist_ok=True, parents=True)

for i in tqdm.tqdm(range(num_frames)):
    img_path = str(subject_path / ims[i]["ims"][cam_id])
    img = cv2.imread(img_path)
    if distort:
        img = cv2.undistort(img, K, D)

    H, W = int(img.shape[0] * ratio), int(img.shape[1] * ratio)
    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)

    vertices_path = vertices_dir / f"{i}.npy"
    vertices = np.load(vertices_path)
    
    rendered_img = render_smpl(vertices, img, K, R, T)

    import pdb;pdb.set_trace()
    cv2.imwrite(f"tmp/subject_id_{subject_id}-distort_{distort}-{i}.png", rendered_img)
    cv2.imshow('main', rendered_img)
    cv2.waitKey(50) & 0xFF
