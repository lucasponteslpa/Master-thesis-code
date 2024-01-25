from mesh_utils import write_obj
from background_verts_module import back_verts_generator, back_faces_generator
from foreground_mesh_verts_module import pforeground_mesh_verts, pforeground_mesh_faces
from merge_mesh_verts_module import pmerge_mesh_verts, pmerge_mesh_faces
import numpy as np
from halide import imageio
import cv2

total_time = 0.0

canny_buf = imageio.imread("canny.png")[:,:].astype(np.uint8)
assert canny_buf.max() == 255
assert canny_buf.ndim == 2
assert canny_buf.dtype == np.uint8

back_canny_buf = imageio.imread("back_canny.png")[:,:].astype(np.uint8)
back_canny_buf = back_canny_buf.astype(np.uint8)
assert back_canny_buf.max() == 255
assert back_canny_buf.ndim == 2
assert back_canny_buf.dtype == np.uint8

depth_buf = imageio.imread("depth_quant.png")[0,:,:].astype(np.uint8)
assert depth_buf.ndim == 2
assert depth_buf.dtype == np.uint8


###########################################
###### BACKGROUND MESH GENERATOR ##########
###########################################

back_tests = {
        "back_verts": back_verts_generator,
        "back_faces": back_faces_generator
    }

IB_shape = (( canny_buf.shape[0]//16) + 1, (canny_buf.shape[1]//16) + 1)
IB_verts_shape = (3, (canny_buf.shape[0]//16 ) + 1, (canny_buf.shape[1]//16) + 1)
P_faces_shape = (3, ((canny_buf.shape[0]//16))*((canny_buf.shape[1]//16))*2)
IB_merge_mask = np.empty(IB_shape, dtype=canny_buf.dtype)
IB_verts_buf = np.empty(IB_verts_shape, dtype=np.float32)
back_faces_buf = np.empty(P_faces_shape, dtype=np.int32)
I_labels = np.empty(canny_buf.shape, dtype=canny_buf.dtype)

back_verts_shape = (3, ((canny_buf.shape[0]//16)+1)*((canny_buf.shape[1]//16)+1))
back_verts_buf = np.empty(back_verts_shape, dtype=np.float32)
I_back_verts_idx = np.empty(canny_buf.shape, dtype=np.int32)

import timeit
timing_iterations = 10
for name, fn in back_tests.items():
        print("Running %s... " % name, end="")
        if name == "back_verts":
                t = timeit.Timer(lambda: fn(canny_buf, back_canny_buf, depth_buf, canny_buf.shape[1]//16, canny_buf.shape[0]//16 , IB_merge_mask, IB_verts_buf, I_labels))
                avg_time_sec = t.timeit(number=timing_iterations) / timing_iterations
                total_time += avg_time_sec
                print("time: %fms" % (avg_time_sec * 1e3))
        else:
        #      pass
                t = timeit.Timer(lambda: fn(IB_verts_buf, back_verts_buf, back_faces_buf, I_back_verts_idx))
                avg_time_sec = t.timeit(number=timing_iterations) / timing_iterations
                total_time += avg_time_sec
                print("time: %fms" % (avg_time_sec * 1e3))

back_faces = back_faces_buf.transpose((1,0))
back_faces = back_faces[np.where((back_faces<0).sum(axis=-1)==0)]
back_verts = back_verts_buf.transpose((1,0))
back_verts[:,-1] = (back_verts[:,-1]/255)

###########################################
###### FOREGROUND MESH GENERATOR ##########
###########################################

tests = {
        "fore_verts": pforeground_mesh_verts,
        "fore_faces": pforeground_mesh_faces
    }

# create a Buffer-compatible object for the output; we'll use np.array

out_f_shape = (((canny_buf.shape[0]//16)*(canny_buf.shape[1]//16)),2*17*17,3)
out_Nf_shape = ((canny_buf.shape[0]//16)*(canny_buf.shape[1]//16))
out_limg_shape = (canny_buf.shape[0],canny_buf.shape[1])
P_faces_buf = np.zeros(out_f_shape, dtype=np.int32)
P_Nfaces_buf = np.zeros(out_Nf_shape, dtype=np.int32)
I_Vlabels_buf = np.zeros(out_limg_shape, dtype=np.int32)
I_Vidx_buf = np.zeros(out_limg_shape, dtype=np.int32) - 1

import timeit
timing_iterations = 10
for name, fn in tests.items():
        print("Running %s... " % name, end="")
        if name == "fore_verts":
            t = timeit.Timer(lambda: fn(canny_buf, back_canny_buf, depth_buf, P_faces_buf, P_Nfaces_buf, I_Vlabels_buf))
            avg_time_sec = t.timeit(number=timing_iterations) / timing_iterations
            total_time += avg_time_sec
            print("time: %fms" % (avg_time_sec * 1e3))
        else:
            NV = (I_Vlabels_buf>0).sum()
            NF = P_Nfaces_buf.sum()
            vert_buf = np.zeros((NV,3), dtype=np.float32)
            face_buf = np.zeros((NF,3), dtype=np.int32)
            t = timeit.Timer(lambda: fn(depth_buf, P_faces_buf, I_Vlabels_buf, back_verts_shape[1], I_Vidx_buf, vert_buf, face_buf))
            avg_time_sec = t.timeit(number=timing_iterations) / timing_iterations
            total_time += avg_time_sec
            print("time: %fms" % (avg_time_sec * 1e3))


###########################################
######### MERGE MESH GENERATOR ############
###########################################
tests = {
        "merge_verts": pmerge_mesh_verts,
        "merge_faces": pmerge_mesh_faces
    }
merge_out_limg_buf = np.zeros(out_limg_shape, dtype=np.int32)
merge_out_f_buf = np.zeros(out_f_shape, dtype=np.int32) - 1
merge_out_Nf_buf = np.zeros(out_Nf_shape, dtype=np.int32)

import timeit
timing_iterations = 10
for name, fn in tests.items():
        print("Running %s... " % name, end="")
        if name == "merge_verts":
                t = timeit.Timer(lambda: fn(I_Vidx_buf, I_back_verts_idx, I_labels, IB_merge_mask, NV+back_verts_shape[1], merge_out_f_buf, merge_out_Nf_buf, merge_out_limg_buf))
                avg_time_sec = t.timeit(number=timing_iterations) / timing_iterations
                total_time += avg_time_sec
                print("time: %fms" % (avg_time_sec * 1e3))
        else:
                NV = (merge_out_limg_buf>0).sum()
                NF = merge_out_Nf_buf.sum()
                merge_vert_buf = np.zeros((NV,3), dtype=np.float32)
                merge_uv_buf = np.zeros((NV,3), dtype=np.float32)
                merge_face_buf = np.zeros((NF,3), dtype=np.int32)
                t = timeit.Timer(lambda: fn(depth_buf, merge_out_f_buf, merge_out_limg_buf, merge_vert_buf, merge_uv_buf, merge_face_buf))
                avg_time_sec = t.timeit(number=timing_iterations) / timing_iterations
                total_time += avg_time_sec
                print("time: %fms" % (avg_time_sec * 1e3))

print("Total time: %fms" % (total_time * 1e3))
all_verts = np.concatenate((back_verts, vert_buf, merge_vert_buf), axis=0)
all_faces = np.concatenate((back_faces, face_buf, merge_face_buf), axis=0)
write_obj('', v_pos=all_verts,t_pos_idx=all_faces)
cv2.imwrite('mid_l.png', 255*((merge_out_limg_buf - merge_out_limg_buf.min())/(merge_out_limg_buf.max()-merge_out_limg_buf.min())))