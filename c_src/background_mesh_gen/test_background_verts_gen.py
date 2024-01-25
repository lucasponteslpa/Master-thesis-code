from background_verts_module import back_verts_generator, back_faces_generator
from halide import imageio
import numpy as np
import cv2
import os

def write_obj(folder,
              v_pos=None,
              v_nrm=None,
              v_tex=None,
              t_pos_idx=None,
              t_nrm_idx=None,
              t_tex_idx=None,
              save_material=True,
              file_name = 'mesh.obj'):
    obj_file = os.path.join(folder, file_name)
    print("Writing mesh: ", obj_file)
    with open(obj_file, "w") as f:
        f.write("mtllib mesh.mtl\n")
        f.write("g default\n")

        v_pos = v_pos if v_pos is not None else None
        v_nrm = v_nrm if v_nrm is not None else None
        v_tex = v_tex if v_tex is not None else None

        t_pos_idx = t_pos_idx if t_pos_idx is not None else None
        t_nrm_idx = t_nrm_idx if t_nrm_idx is not None else None
        t_tex_idx = t_tex_idx if t_tex_idx is not None else None

        print("    writing %d vertices" % len(v_pos))
        for v in v_pos:
            f.write('v {} {} {} \n'.format(v[0], v[1], v[2]))

        if v_tex is not None:
            print("    writing %d texcoords" % len(v_tex))
            assert(len(t_pos_idx) == len(t_tex_idx))
            for v in v_tex:
                f.write('vt {} {} \n'.format(v[0], 1.0 - v[1]))

        # if v_nrm is not None:
        #     print("    writing %d normals" % len(v_nrm))
        #     assert(len(t_pos_idx) == len(t_nrm_idx))
        #     for v in v_nrm:
        #         f.write('vn {} {} {}\n'.format(v[0], v[1], v[2]))

        # faces
        f.write("s 1 \n")
        f.write("g pMesh1\n")
        f.write("usemtl defaultMat\n")

        # Write faces
        print("    writing %d faces" % len(t_pos_idx))
        for i in range(len(t_pos_idx)):
            f.write("f ")
            for j in range(3):
                f.write(' %s/%s/%s' % (str(t_pos_idx[i][j]+1), '' if v_tex is None else str(t_tex_idx[i][j]+1), '' if v_nrm is None else str(t_nrm_idx[i][j]+1)))
            f.write("\n")


    print("Done exporting mesh")

# Read in some file for input
canny_buf = imageio.imread("canny.png")[:,:]
assert canny_buf.max() == 255
assert canny_buf.ndim == 2
assert canny_buf.dtype == np.uint8

back_canny_buf = imageio.imread("back_canny.png")[:,:]
assert back_canny_buf.max() == 255
assert back_canny_buf.ndim == 2
assert back_canny_buf.dtype == np.uint8

depth_buf = imageio.imread("depth_quant.png")[0,:,:]
assert depth_buf.ndim == 2
assert depth_buf.dtype == np.uint8

tests = {
        "verts": back_verts_generator,
        "faces": back_faces_generator
    }

IB_shape = (( canny_buf.shape[0]//16) + 1, (canny_buf.shape[1]//16) + 1)
IB_verts_shape = (3, (canny_buf.shape[0]//16 ) + 1, (canny_buf.shape[1]//16) + 1)
IB_uvs_shape = (3, (canny_buf.shape[0]//16 ) + 1, (canny_buf.shape[1]//16) + 1)
P_faces_shape = (3, ((canny_buf.shape[0]//16))*((canny_buf.shape[1]//16))*2)
IB_merge_mask = np.empty(IB_shape, dtype=canny_buf.dtype)
IB_verts_buf = np.empty(IB_verts_shape, dtype=np.float32)
IB_fore_verts_buf = np.empty(IB_verts_shape, dtype=np.float32)
IB_uvs_buf = np.empty(IB_uvs_shape, dtype=np.float32)
back_faces_buf = np.empty(P_faces_shape, dtype=np.int32)
I_labels = np.empty(canny_buf.shape, dtype=canny_buf.dtype)

back_verts_shape = (3, 2*((canny_buf.shape[0]//16)+1)*((canny_buf.shape[1]//16)+1))
back_uvs_shape = (3, 2*((canny_buf.shape[0]//16)+1)*((canny_buf.shape[1]//16)+1))
back_verts_buf = np.empty(back_verts_shape, dtype=np.float32)
back_uvs_buf = np.empty(back_uvs_shape, dtype=np.float32)
I_back_verts_idx = np.empty(canny_buf.shape, dtype=np.int32)

import timeit
timing_iterations = 10
for name, fn in tests.items():
        print("Running %s... " % name, end="")
        if name == "verts":
                t = timeit.Timer(lambda: fn(canny_buf, back_canny_buf, depth_buf, canny_buf.shape[1]//16, canny_buf.shape[0]//16 , IB_merge_mask, IB_verts_buf,IB_fore_verts_buf,IB_uvs_buf, I_labels))
                avg_time_sec = t.timeit(number=timing_iterations) / timing_iterations
                print("time: %fms" % (avg_time_sec * 1e3))
        else:
        #      pass
                t = timeit.Timer(lambda: fn(IB_verts_buf, IB_fore_verts_buf, IB_uvs_buf, I_labels, back_verts_buf, back_uvs_buf, back_faces_buf, I_back_verts_idx))
                avg_time_sec = t.timeit(number=timing_iterations) / timing_iterations
                print("time: %fms" % (avg_time_sec * 1e3))

faces = back_faces_buf.transpose((1,0))
faces = faces[np.where((faces<0).sum(axis=-1)==0)]
verts = back_verts_buf.transpose((1,0))
verts[:,-1] = 1.0 - (verts[:,-1]/255)
write_obj('', v_pos=verts,t_pos_idx=faces)
breakpoint()
imageio.imwrite("img_idx.png", 255*(I_back_verts_idx/I_back_verts_idx.max()))
imageio.imwrite("verts_labels.png", I_labels[:,:])
block_mask = 255*IB_merge_mask[1,:,:]
block_mask = cv2.resize(block_mask, dsize=(I_labels.shape), interpolation=cv2.INTER_NEAREST)
imageio.imwrite("block_mask.png", block_mask)
