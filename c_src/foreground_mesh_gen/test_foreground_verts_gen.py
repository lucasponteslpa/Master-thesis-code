from foreground_mesh_verts_module import pforeground_mesh_verts, pforeground_mesh_faces
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
canny_buf = imageio.imread("canny.png")[:,:].astype(np.uint8)
# canny_buf = np.arange(0,canny_buf.shape[0]*canny_buf.shape[1]).astype(np.uint8).reshape(canny_buf.shape)
# print(canny_buf[:17,:17])
# assert canny_buf.max() == 255
assert canny_buf.ndim == 2
# assert canny_buf.dtype == np.uint8

back_canny_buf = imageio.imread("back_canny.png")[:,:].astype(np.uint8)
# back_canny_buf = np.arange(0,back_canny_buf.shape[0]*back_canny_buf.shape[1]).astype(np.uint8).reshape(back_canny_buf.shape)
# back_canny_buf[back_canny_buf>255] = 255
# back_canny_buf = back_canny_buf.astype(np.uint8)
# assert back_canny_buf.max() == 255
assert back_canny_buf.ndim == 2
# assert back_canny_buf.dtype == np.uint8

depth_buf = imageio.imread("depth_quant.png")[0,:,:].astype(np.uint8)
# print(depth_buf[:17,:17])
print(depth_buf.shape)
# assert depth_buf.ndim == 2
# assert depth_buf.dtype == np.uint8
tests = {
        "verts": pforeground_mesh_verts,
        "faces": pforeground_mesh_faces
    }

# create a Buffer-compatible object for the output; we'll use np.array

out_f_shape = (((canny_buf.shape[0]//16)*(canny_buf.shape[1]//16)),2*17*17,3)
out_Nf_shape = ((canny_buf.shape[0]//16)*(canny_buf.shape[1]//16))
out_limg_shape = (canny_buf.shape[0],canny_buf.shape[1])
P_faces_buf = np.zeros(out_f_shape, dtype=np.int32)
P_Nfaces_buf = np.zeros(out_Nf_shape, dtype=np.int32)
I_Vlabels_buf = np.zeros(out_limg_shape, dtype=np.int32)
I_Vidx_buf = np.zeros(out_limg_shape, dtype=np.int32) - 1
I_ForeMask_buf = np.zeros(out_limg_shape, dtype=np.int32)

import timeit
timing_iterations = 10
for name, fn in tests.items():
        print("Running %s... " % name, end="")
        if name == "verts":
            t = timeit.Timer(lambda: fn(canny_buf, back_canny_buf, depth_buf, P_faces_buf, P_Nfaces_buf, I_Vlabels_buf, I_ForeMask_buf))
            avg_time_sec = t.timeit(number=timing_iterations) / timing_iterations
            print("time: %fms" % (avg_time_sec * 1e3))
        else:
            NV = (I_Vlabels_buf>0).sum()
            NF = P_Nfaces_buf.sum()
            vert_buf = np.zeros((NV,3), dtype=np.float32)
            uv_buf = np.zeros((NV,3), dtype=np.float32)
            face_buf = np.zeros((NF,3), dtype=np.int32)
            t = timeit.Timer(lambda: fn(depth_buf, P_faces_buf, I_Vlabels_buf, 0, I_Vidx_buf, vert_buf, uv_buf, face_buf))
            avg_time_sec = t.timeit(number=timing_iterations) / timing_iterations
            print("time: %fms" % (avg_time_sec * 1e3))

cv2.imwrite("fore_l.png", 255*(I_Vlabels_buf/I_Vlabels_buf.max()))
cv2.imwrite("fore_mask.png", 255*(I_ForeMask_buf/I_ForeMask_buf.max()))
cv2.imwrite("fore_idx.png", 255*((I_Vidx_buf - I_Vidx_buf.min())/(I_Vlabels_buf.max() - I_Vidx_buf.min())))

vert_buf[:,-1] = 1 - vert_buf[:,-1]
write_obj('', v_pos=vert_buf,t_pos_idx=face_buf[(face_buf!=-1).all(axis=1)])
# breakpoint()
print(P_Nfaces_buf.max())
