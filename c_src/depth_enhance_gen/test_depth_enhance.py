import numpy as np
import cv2
from halide import imageio
from depth_enhance_module import depth_enhance_gen

masks = (np.load('seg_stylist.npy')).astype(np.uint8)
depth = imageio.imread("depth_inference.png")[0,:,:]
out_depth = np.zeros(depth.shape, dtype=np.float32)
sum_kernel = np.zeros(masks.shape, dtype=np.float32)
blocks_depth = np.zeros(((depth.shape[0]//32)*(depth.shape[1]//32), 32, 32), dtype=np.float32)
mask_sum_c = masks.sum(axis=0)
mask_inv = 1 - mask_sum_c/mask_sum_c.max()


mask_areas = (masks.sum(axis=-1)).sum(axis=-1)
masks_area_num = masks*mask_areas.reshape(masks.shape[0],1,1)
masks_union = np.zeros(masks.shape, dtype=np.uint8)
for c in range(masks.shape[0]):
    m_s = (masks*masks[c,:,:].reshape(1,masks.shape[1], masks.shape[2]))
    m_s = (m_s.sum(axis=-1)).sum(axis=-1)
    m_s[c]=0
    if m_s.max()>1:
        m_i = np.argmax(m_s)
        masks_union[c] = ((masks[c]>0)+(masks[m_i])>0).astype(np.uint8)
    else:
        masks_union[c] = masks[c]

import timeit
timing_iterations = 10
t = timeit.Timer(lambda: depth_enhance_gen(depth, masks_union, out_depth, sum_kernel, blocks_depth))
avg_time_sec = t.timeit(number=timing_iterations) / timing_iterations
print("time: %fms" % (avg_time_sec * 1e3))

depth_enhance_gen(depth, masks_union, out_depth, sum_kernel, blocks_depth)
dst = cv2.inpaint((255*out_depth/out_depth.max()).astype(np.uint8),mask_inv.astype(np.uint8),3,cv2.INPAINT_TELEA)

kernel = np.ones((3,3))
dilated_depth = cv2.dilate(dst,kernel,iterations=1)
# depth_blur = cv2.GaussianBlur(dilated_depth.astype(np.uint8), (3,3),0)
img_depth_canny = cv2.Canny(dilated_depth, 30, 50, L2gradient=True)

dilated_d = cv2.dilate(dilated_depth,kernel,iterations=1)
dilated_depth[img_depth_canny==255] = dilated_d[img_depth_canny==255]
dilated_d = cv2.dilate(dilated_depth,kernel,iterations=2)
sharp_depth = cv2.erode(dilated_d,kernel,iterations=2)
# dilated_depth = cv2.erode(dilated_depth,kernel,iterations=1)

cv2.imwrite("sum_kernel.png",sum_kernel.sum(axis=0)/(masks_union.sum(axis=0)))
cv2.imwrite("masks_union.png",255*(masks_union.sum(axis=0)/masks_union.sum(axis=0).max()))
cv2.imwrite("masks.png",255*(masks.sum(axis=0)/masks.max()))
cv2.imwrite('result.png', 255*out_depth/out_depth.max())
cv2.imwrite('result_interp.png', sharp_depth)
cv2.imwrite('depth_canny.png', img_depth_canny)
