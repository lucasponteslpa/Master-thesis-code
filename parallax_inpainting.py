from renderTexture import TextureRenderer
import numpy as np
import cv2
import os
import torch
import argparse
from mesh_utils import *
from utilities import gen_video
from depth_inference import midas_inference
from simple_lama_inpainting import SimpleLama
# from background_mesh import BackgroundMesh
# from foreground_mesh import ForegroundMesh
# from quadtrees import mQuadT, Block
# from MiDaS.depth_estimation import runMidas

from background_verts_module import background_mesh_verts, background_mesh_faces
from foreground_mesh_verts_module import pforeground_mesh_verts, pforeground_mesh_faces
from merge_mesh_verts_module import pmerge_mesh_verts, pmerge_mesh_faces

class ParallaxInpainting:
    def __init__(self,
                 rgb_dir,
                 inpaint_dir,
                 depth_dir,
                 depth_max_dim,
                 render_res=720,
                 block_size = 16,
                 midas_mobile=True):
        self.rgb_dir = rgb_dir
        self.inpaint_dir = inpaint_dir
        self. depth_dir = depth_dir
        self.depth_max_dim = depth_max_dim
        self.render_res = render_res
        self.block_size = block_size
        self.blocks_per_dim = depth_max_dim//block_size
        self.midas_mobile = midas_mobile


    def load_depth_and_canny(self, debug=False):
        if not self.depth_dir is None:
            img_depth = cv2.imread(self.depth_dir)
        else:
            color = cv2.imread(self.rgb_dir)
            color = cv2.cvtColor(color,cv2.COLOR_BGR2RGB)/255.0
        img_depth = midas_inference(self.rgb_dir,out_path='depth_inference.png')
        img_depth = np.repeat(img_depth, 3, axis=-1)

        DEPTH_MAX = img_depth.max()
        img_depth = img_depth/img_depth.max()

        width =  img_depth.shape[1]
        height = img_depth.shape[0]

        ratio_w = 1.0 if width >= height else width/height
        ratio_h = 1.0 if height >= width else height/width

        self.width =  np.round(self.blocks_per_dim*self.block_size*ratio_w).astype(int)
        self.height = np.round(self.blocks_per_dim*self.block_size*ratio_h).astype(int)
        self.width += (self.block_size - self.width%self.block_size)
        self.height += (self.block_size - self.height%self.block_size)

        self.ratio_w = 1.0 if self.width >=  self.height else self.width/self.height
        self.ratio_h = 1.0 if self.height >= self.width else  self.height/self.width

        dim = (self.width, self.height)
        color_blur = cv2.GaussianBlur((255*color).astype(np.uint8), (7,7),0)
        self.color_res = cv2.resize(cv2.cvtColor(color_blur,cv2.COLOR_RGB2BGR), dsize=(3*self.width, 3*self.height), interpolation = cv2.INTER_CUBIC)
        cv2.imwrite("color.png", self.color_res)
        depth_res_c = cv2.resize((img_depth*255).astype(np.uint8), dsize=dim, interpolation = cv2.INTER_CUBIC)
        cv2.imwrite("depth_inference.png", depth_res_c)
        kernel = np.ones((3, 3), 'uint8')
        b_d  = cv2.erode(depth_res_c, kernel, iterations=2)
        b_d = cv2.GaussianBlur(b_d, (3,3),0)
        depth_res_c = b_d
        img_depth_canny = cv2.Canny(b_d, 30, 50, L2gradient=True)
        pre_filter_canny = img_depth_canny
        img_depth_canny = self.filter_canny(img_depth_canny)


        if debug:
            cv2.imwrite('prefilter_canny.png', pre_filter_canny)
            cv2.imwrite('canny.png', img_depth_canny)


        dilated_canny = cv2.dilate(img_depth_canny, kernel, iterations=1)
        cv2.imwrite('dilated_canny.png', dilated_canny)
        # sharp_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        # depth_sharp = cv2.filter2D(dilated_canny, -1, sharp_kernel)
        # cv2.imwrite('depth_sharp.png', depth_sharp)

        dilated_d = cv2.dilate(depth_res_c, kernel, iterations=2)
        erode_d = cv2.erode(depth_res_c, kernel, iterations=2)
        depth_res_c[dilated_canny==255] = erode_d[dilated_canny==255]
        depth_res_c[img_depth_canny==255] = dilated_d[img_depth_canny==255]
        dilated_d = cv2.dilate(depth_res_c, kernel, iterations=1)
        erode_d  = cv2.erode(dilated_d, kernel, iterations=1)
        depth_res_c = erode_d

        back_canny = self.get_background_canny(depth_res_c,img_depth_canny, dilated_canny)
        self.back_canny = self.filter_canny(back_canny)
        cv2.imwrite('back_canny.png', self.back_canny)

        # d_back, d_fore = self.custom_dilate(self.back_canny,img_depth_canny, iter=2)
        if debug:
            cv2.imwrite('depth_quant.png', depth_res_c)
            # cv2.imwrite('back_dilated.png', 255*d_back)
            # cv2.imwrite('fore_dilated.png', 255*d_fore)


        img_depth = depth_res_c[::self.block_size,::self.block_size,:]/255
        cv2.imwrite("block_depth.png", 255*img_depth)
        self.img_depth_vert = img_depth
        self.img_depth = depth_res_c
        self.canny = img_depth_canny
        # self.back_canny = back_canny
        print(self.img_depth_vert.shape[0],self.img_depth_vert.shape[1])


    def custom_dilate(self, back_canny, canny, iter=2, k_size=3):
        kernel = np.ones((k_size, k_size), 'uint8')
        back = (back_canny==255).astype(np.uint8)
        fore = (canny==255).astype(np.uint8)
        for _ in range(iter):
            dilate_back = cv2.dilate(back, kernel, iterations=1)
            dilate_fore = cv2.dilate(fore, kernel, iterations=1)
            back = dilate_back*(1-dilate_fore)
            fore = dilate_fore*(1-dilate_back)
        return back, fore


    def filter_canny(self, canny):
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(canny, 15, cv2.CV_32S)
        filtered_canny = np.zeros_like(canny)
        n_pixels = np.prod(canny.shape)
        clip = 50
        for i in range(1,n_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if (area >= clip):
                current = (labels == i).astype("uint8")
                filtered_canny = cv2.bitwise_or(filtered_canny, current)

        return 255*filtered_canny


    def print_ccl(self, labels,n_labels, name='ccl_debug'):
        img = np.zeros((labels.shape[0], labels.shape[1], 3))
        div = n_labels/2
        zero_mask = labels != 0
        for i in range(2):
            aux0 = np.zeros_like(labels)
            aux1 = np.zeros_like(labels)
            mask = labels < div*(i+1)
            aux0[mask*zero_mask] = 255*(((div*(i+1)) - labels[mask*zero_mask])/(div*(i+1)))
            aux1[mask*zero_mask] = 255*(labels[zero_mask*mask]/(div*(i+1)))
            img[:,:,i] = aux0
            img[:,:,i+1] = aux1
        cv2.imwrite(name+'.png', img)
        cv2.imwrite(name+'_region.png', img[64:96+1,448:480+1])


    def get_ccl(self, canny, debug=False):
        n_labels, labels, _, _ = cv2.connectedComponentsWithStats(canny, 15, cv2.CV_32S)
        if debug:
            self.print_ccl(labels, n_labels)
        return n_labels, labels


    def get_background_canny(self, depth, canny, dilated_canny):
        c_d_diff = dilated_canny - canny
        d_canny_locals = np.where(c_d_diff==255)
        back_canny = np.zeros_like(canny)

        for l_idx in zip(d_canny_locals[0], d_canny_locals[1]):
            if l_idx[0]>0 and l_idx[0] < (canny.shape[0]-1) and l_idx[1]>0 and l_idx[1] < (canny.shape[1]-1):
                depth_region = depth[l_idx[0]-1:l_idx[0]+2, l_idx[1]-1:l_idx[1]+2]
                canny_region = canny[l_idx[0]-1:l_idx[0]+2, l_idx[1]-1:l_idx[1]+2]
                canny_locals = np.where(canny_region==255)
                c_l_depth = depth_region[canny_locals]
                diff = np.abs(c_l_depth/255 - depth[l_idx]/255)
                if np.array([diff]).max() > 0.025:
                    # breakpoint()
                    back_canny[l_idx] = 255
        return back_canny


    def get_background_ccl(self, depth, ccl, dilated_ccl):
        c_d_diff = dilated_ccl - ccl
        d_ccl_locals = np.where(c_d_diff>0)
        back_ccl = np.zeros_like(ccl)

        for l_idx in zip(d_ccl_locals[0], d_ccl_locals[1]):
            if l_idx[0]>0 and l_idx[0] < (ccl.shape[0]-1) and l_idx[1]>0 and l_idx[1] < (ccl.shape[1]-1):
                depth_region = depth[l_idx[0]-1:l_idx[0]+2, l_idx[1]-1:l_idx[1]+2]
                ccl_region = ccl[l_idx[0]-1:l_idx[0]+2, l_idx[1]-1:l_idx[1]+2]
                ccl_locals = np.where(ccl_region>0)
                c_l_depth = depth_region[ccl_locals]
                diff = np.abs(c_l_depth/255 - depth[l_idx]/255)
                if np.array([diff]).max() > 0.05:
                    # breakpoint()
                    back_ccl[l_idx] = -1*dilated_ccl[l_idx]
        return back_ccl


    def get_ccl_means(self,n_ccl, ccl, back_ccl, depth):
        ccl_means = np.zeros((n_ccl,3))
        for i in range(1,n_ccl):
            cc_locals = np.where(ccl==i)
            cc_back_locals = np.where(back_ccl==(-i))
            ccl_means[i,0] = i
            ccl_means[i,1] = depth[cc_locals].mean()
            ccl_means[i,2] = depth[cc_back_locals].mean()
        return ccl_means


        # breakpoint()

    def halide_mesh(self, get_screenshot = False):
        self.load_depth_and_canny(debug=True)
        canny_buf = self.canny
        assert canny_buf.max() == 255
        assert canny_buf.ndim == 2
        assert canny_buf.dtype == np.uint8

        back_canny_buf = self.back_canny
        assert back_canny_buf.max() == 255
        assert back_canny_buf.ndim == 2
        assert back_canny_buf.dtype == np.uint8

        depth_buf = (self.img_depth[:,:,0]).astype(np.uint8)
        assert depth_buf.ndim == 2
        assert depth_buf.dtype == np.uint8

        IB_shape = (( canny_buf.shape[0]//self.block_size) + 1, (canny_buf.shape[1]//self.block_size) + 1)
        IB_verts_shape = (3, (canny_buf.shape[0]//self.block_size ) + 1, (canny_buf.shape[1]//self.block_size) + 1)
        IB_uvs_shape = (3, (canny_buf.shape[0]//self.block_size ) + 1, (canny_buf.shape[1]//self.block_size) + 1)
        P_faces_shape = (3, ((canny_buf.shape[0]//self.block_size))*((canny_buf.shape[1]//self.block_size))*2)
        IB_merge_mask = np.empty(IB_shape, dtype=canny_buf.dtype)
        IB_verts_buf = np.empty(IB_verts_shape, dtype=np.float32)
        IB_fore_verts_buf = np.empty(IB_verts_shape, dtype=np.float32)
        IB_uvs_buf = np.empty(IB_uvs_shape, dtype=np.float32)
        back_faces_buf = np.empty(P_faces_shape, dtype=np.int32)
        I_labels = np.empty(canny_buf.shape, dtype=canny_buf.dtype)

        back_verts_shape = (3, 2*((canny_buf.shape[0]//self.block_size)+1)*((canny_buf.shape[1]//self.block_size)+1))
        back_uvs_shape = (3, 2*((canny_buf.shape[0]//self.block_size)+1)*((canny_buf.shape[1]//self.block_size)+1))
        back_verts_buf = np.empty(back_verts_shape, dtype=np.float32)
        back_uvs_buf = np.empty(back_uvs_shape, dtype=np.float32)
        I_back_verts_idx = np.empty(canny_buf.shape, dtype=np.int32)

        background_mesh_verts(canny_buf, back_canny_buf, depth_buf, self.block_size, canny_buf.shape[1]//self.block_size, canny_buf.shape[0]//self.block_size , IB_merge_mask, IB_verts_buf,IB_fore_verts_buf,IB_uvs_buf, I_labels)
        background_mesh_faces(IB_verts_buf, IB_fore_verts_buf, IB_uvs_buf, I_labels, self.block_size, back_verts_buf, back_uvs_buf, back_faces_buf, I_back_verts_idx)
        cv2.imwrite('v_labels.png', 255*(I_labels/I_labels.max()))
        mask_res = 255*(cv2.resize(IB_merge_mask[:-1,:-1], dsize=(I_labels.shape[0], I_labels.shape[1]), interpolation=cv2.INTER_NEAREST )>0).astype(np.int32)
        cv2.imwrite('mask.png', 255*(mask_res/mask_res.max()))
        faces = back_faces_buf.transpose((1,0))
        faces = faces[np.where((faces<0).sum(axis=-1)==0)]
        back_verts = back_verts_buf.transpose((1,0))
        # back_verts[:,-1] = (back_verts[:,-1]/255)
        back_uvs = back_uvs_buf.transpose((1,0))
        # uvs[:,-1] = 1.0 - (verts[:,-1]/255)

        out_f_shape = (((canny_buf.shape[0]//self.block_size)*(canny_buf.shape[1]//self.block_size)),2*(self.block_size+1)*(self.block_size+1),3)
        out_Nf_shape = ((canny_buf.shape[0]//self.block_size)*(canny_buf.shape[1]//self.block_size))
        out_limg_shape = (canny_buf.shape[0],canny_buf.shape[1])
        P_faces_buf = np.zeros(out_f_shape, dtype=np.int32)
        P_Nfaces_buf = np.zeros(out_Nf_shape, dtype=np.int32)
        I_Vlabels_buf = np.zeros(out_limg_shape, dtype=np.int32)
        I_Vidx_buf = np.zeros(out_limg_shape, dtype=np.int32) - 1
        I_ForeMask_buf = np.zeros(out_limg_shape, dtype=np.int32)

        pforeground_mesh_verts(canny_buf, back_canny_buf, depth_buf, self.block_size, P_faces_buf, P_Nfaces_buf, I_Vlabels_buf, I_ForeMask_buf)

        NV = (I_Vlabels_buf>0).sum()
        NF = P_Nfaces_buf.sum()
        fore_vert_buf = np.zeros((NV,3), dtype=np.float32)
        fore_uv_buf = np.zeros((NV,3), dtype=np.float32)
        fore_face_buf = np.zeros((NF,3), dtype=np.int32)
        pforeground_mesh_faces(depth_buf, P_faces_buf, I_Vlabels_buf, back_verts_shape[1], self.block_size, I_Vidx_buf, fore_vert_buf, fore_uv_buf, fore_face_buf)

        merge_out_limg_buf = np.zeros(out_limg_shape, dtype=np.int32)
        merge_out_f_buf = np.zeros(out_f_shape, dtype=np.int32) - 1
        merge_out_Nf_buf = np.zeros(out_Nf_shape, dtype=np.int32)

        pmerge_mesh_verts(I_Vidx_buf, I_back_verts_idx, I_labels, IB_merge_mask, NV+back_verts_shape[1], self.block_size, merge_out_f_buf, merge_out_Nf_buf, merge_out_limg_buf)
        NV = (merge_out_limg_buf>0).sum()
        NF = merge_out_Nf_buf.sum()
        merge_vert_buf = np.zeros((NV,3), dtype=np.float32)
        merge_uv_buf = np.zeros((NV,3), dtype=np.float32)
        merge_face_buf = np.zeros((NF,3), dtype=np.int32)

        pmerge_mesh_faces(depth_buf, merge_out_f_buf, merge_out_limg_buf, self.block_size, merge_vert_buf, merge_uv_buf, merge_face_buf)
        all_verts = np.concatenate((back_verts, fore_vert_buf, merge_vert_buf), axis=0)
        all_verts[:,-1] = 1.0 - all_verts[:,-1]
        all_uvs = np.concatenate((back_uvs[:,:2], fore_uv_buf[:,:2], merge_uv_buf[:,:2]), axis=0)
        all_faces = np.concatenate((fore_face_buf, merge_face_buf), axis=0)
        total_faces = np.concatenate((faces, all_faces), axis=0)
        write_obj('',
                    v_pos=torch.from_numpy(all_verts),
                    t_pos_idx=torch.from_numpy(total_faces),
                    file_name='total_mesh.obj')
        # inpaint_func = SimpleLama()
        # self.inpaint_dir = 'images/inpaint_curr.png'
        # res_inp = inpaint_func(self.color_res, mask_res+(255*(I_ForeMask_buf/I_ForeMask_buf.max())))
        # cv2.imwrite(self.inpaint_dir, res_inp)
        render = TextureRenderer(
                             height=self.render_res,
                             width= self.render_res,
                             vertices=all_verts.reshape(-1),
                             uv_coords=all_uvs.reshape(-1),
                             indices=all_faces.reshape(-1),
                             indices_back=faces.reshape(-1),
                             img_file=self.rgb_dir,
                             img_file_back=self.inpaint_dir,
                             texture_dims=(2*(self.width),2*(self.height)),
                             frames_path='frames_orig')
        render.runCircleZoomWindow(z_init=6, get_screenshot=get_screenshot)
        if get_screenshot:
            gen_video('frames_orig/','movie','sp61_dimas_stylists_parallax_inpaint_v2'+str(self.depth_max_dim)+'b'+str(self.block_size)+'.mp4')


    def renderCircleZoom(self, get_screenshot=False):
        render = TextureRenderer(
                             height=self.render_res,
                             width= self.render_res,
                             vertices=self.foreground_verts.reshape(-1).detach().numpy(),
                             uv_coords=self.foreground_uvs.reshape(-1).detach().numpy(),
                             indices=self.foreground_faces.reshape(-1).numpy(),
                             indices_back=self.background_faces.reshape(-1).numpy(),
                             img_file=self.rgb_dir,
                             img_file_back=self.inpaint_dir,
                             texture_dims=(2*(self.width),2*(self.height)),
                             frames_path='frames_orig')
        render.runCircleZoomWindow(z_init=6, get_screenshot=get_screenshot)
        if get_screenshot:
            gen_video('frames_orig/','movie','sp61_Lmidas_hdrp6_parallax_inpaint_'+str(self.depth_max_dim)+'b'+str(self.block_size)+'.mp4')


    def save_mesh(self, path='mesh/', file_name = 'parallax_mesh.obj', verts=None, faces=None):
        if verts is None or faces is None:
            total_faces = torch.cat([self.background_faces, self.foreground_faces], dim=0)
            write_obj(path,
                    v_pos=self.foreground_verts,
                    t_pos_idx=total_faces,
                    file_name=file_name)
        else:
            write_obj(path,
                    v_pos=verts,
                    t_pos_idx=faces,
                    file_name=file_name)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Cube fit example')
    parser.add_argument('--rgb_dir', help='specify output directory', default='images/Stylists.png')
    parser.add_argument('--inpaint_dir', help='specify output directory', default='images/stylists_inp_640b32_1024.png')
    parser.add_argument('--depth_dir', help='specify output directory', default=None)
    parser.add_argument('--discontinuous', action='store_true', default=False)
    parser.add_argument('--resolution', type=int, default=32, required=False)
    parser.add_argument('--mp4save', action='store_true', default=False)
    args = parser.parse_args()

    parallax = ParallaxInpainting(args.rgb_dir,
                                  args.inpaint_dir,
                                  args.depth_dir,
                                  depth_max_dim=1280,
                                  block_size=32,
                                  render_res=720,
                                  midas_mobile=False)
    # parallax = ParallaxInpainting(args.rgb_dir, None, args.depth_dir, depth_max_dim=640)
    # parallax.run(render_mesh=False,debug=True,get_screenshot=False)
    parallax.halide_mesh(False)