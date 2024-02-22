from rederSLIDEMesh import SLIDEMeshRenderer
import open3d as o3d
import numpy as np
import argparse
import glob
import os

def load_mesh(ply_file):
    mesh = o3d.io.read_triangle_mesh(ply_file)
    faces = np.asarray(mesh.triangles).astype(np.float32)
    verts = np.asarray(mesh.vertices).astype(np.float32)
    colors = np.asarray(mesh.vertex_colors).astype(np.float32)

    return verts, colors, faces

def get_image_file_paths(path) -> list:
    """
    Returns path to all images in images folder.

    Returns:
        image_file_names: List of strings.
    """

    image_file_names = glob.glob(path+"/*")
    return image_file_names

def rendering(ply_file, alphas_file, render_height, render_width, save_frames=False, n_frames=1000, output_path='frames_orig'):
    verts, colors, faces = load_mesh(ply_file)
    alphas = np.expand_dims(np.load(alphas_file), axis=-1)
    alphas = (alphas - alphas.min())/(alphas.max() - alphas.min())
    colors = np.concatenate((colors,alphas), axis=-1).astype(np.float32)
    
    verts[:,0] = 1.0 - (2.0*verts[:,0])
    verts[:,1] = (2.0*verts[:,1]) - 1.0
    verts[:,2] = 1.0 - (verts[:,2])
    
    render = SLIDEMeshRenderer(
                             height=render_height,
                             width=render_width,
                             vertices=verts.reshape(-1),
                             colors=colors.reshape(-1),
                             indices=faces.reshape(-1),
                             frames_path=output_path)
    
    render.runCircleZoomWindow(z_init=-6.0, get_screenshot=save_frames, n_frames=n_frames)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--frames_path", type=str, default="frames_slide", help="Path to PLY file"
    )
    parser.add_argument(
        "--mesh_file", type=str, default="slide_hdrp_subset/meshes/", help="Path to PLY file"
    )

    args = parser.parse_args()

    mesh_path = os.path.join(os.getcwd(), args.mesh_file)
    dir_names = get_image_file_paths(mesh_path)
    for hdrp_dir in dir_names:
        img_paths = get_image_file_paths(hdrp_dir)
        if 'ply' in img_paths[0]:
            mesh_file = img_paths[0]
            alphas_file = img_paths[1]
        else:
            mesh_file = img_paths[1]
            alphas_file = img_paths[0]
        rendering(mesh_file, alphas_file,  1024,720)
