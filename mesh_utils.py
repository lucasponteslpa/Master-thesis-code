import numpy as np
import os
import torch
import cv2
# from quadtrees import  Rect, QuadTree, QuadTreeToMesh

# def gen_region_mesh(depth_res_c,
#                     img_depth_canny,
#                     range_i,
#                     range_j,
#                     blocks_d,
#                     block_idx):
#     # canny_locals = np.where(img_depth_canny[130:146,50:66]==255)
#     canny_region = img_depth_canny[range_i[0]:range_i[1],range_j[0]:range_j[1]]
#     canny_locals = np.where(canny_region==255)

#     # ADD THIS BLOCK IN A PREVIOUS FUNCTION
#     # kernel = np.ones((3, 3), 'uint8')
#     # dilated_d = cv2.dilate(255 - depth_res_c, kernel, iterations=1)
#     # depth_res_c = 255 - depth_res_c
#     # depth_res_c[img_depth_canny==255] = dilated_d[img_depth_canny==255]

#     canny_x = torch.tensor(canny_locals[0])
#     canny_y = torch.tensor(canny_locals[1])
#     canny_verts = torch.cat([canny_y.unsqueeze(-1), canny_x.unsqueeze(-1)], dim=-1)
#     coords = canny_verts.detach().numpy()
#     # points = [Point(*coord) for coord in coords]
#     points = coords
#     width  = canny_region.shape[0]
#     height  = canny_region.shape[1]
#     domain = Rect(width/2, height/2, width, height)
#     qtree = QuadTree(domain, 1)
#     qtree.insert_points(points)
#     depth_region = depth_res_c[range_i[0]:range_i[1]+1,range_j[0]:range_j[1]+1]/255
#     qt_mesh = QuadTreeToMesh(qtree, depth_region,(width+1,height+1,2))
#     # max_v = 1.0/(depth_res_c.shape[0]/(range_i[1]-range_i[0]))
#     qt_mesh.labeling()
#     qt_mesh.get_depth_type()
#     qt_mesh.get_corners_depth()
#     if qt_mesh.Q11 >= blocks_d[block_idx]:
#         blocks_d[block_idx] = qt_mesh.Q11
#     else:
#         qt_mesh.Q11 = blocks_d[block_idx]

#     if qt_mesh.Q12 >= blocks_d[block_idx[0]+1,block_idx[1]]:
#         blocks_d[block_idx[0]+1,block_idx[1]] = qt_mesh.Q12
#     else:
#         qt_mesh.Q12 = blocks_d[block_idx[0]+1,block_idx[1]]

#     if qt_mesh.Q21 >= blocks_d[block_idx[0],block_idx[1]+1]:
#         blocks_d[block_idx[0],block_idx[1]+1] = qt_mesh.Q21
#     else:
#         qt_mesh.Q21 = blocks_d[block_idx[0],block_idx[1]+1]

#     if qt_mesh.Q22 >= blocks_d[block_idx[0]+1,block_idx[1]+1]:
#         blocks_d[block_idx[0]+1,block_idx[1]+1] = qt_mesh.Q22
#     else:
#         qt_mesh.Q22 = blocks_d[block_idx[0]+1,block_idx[1]+1]
#     return qt_mesh
#     # verts, uvs, faces = qt_mesh.create_mesh(max_v)
#     # if len(verts)==4:
#     #     breakpoint()
#     # if len(verts) > 0:
#     #     verts[:,:2] = verts[:,:2] + vtx_ancor[:2]
#     #     uvs = uvs + uv_ancor
#     #     faces = faces + vtx_idx

#     # return verts, uvs, faces

# def gen_mesh_faces(qt_mesh,
#                     max_v,
#                     max_uv,
#                     vtx_ancor,
#                     uv_ancor,
#                     vtx_idx,
#                     blocks_d,
#                     block_idx):

#     qt_mesh.Q11 = blocks_d[block_idx]
#     qt_mesh.Q12 = blocks_d[block_idx[0]+1,block_idx[1]]
#     qt_mesh.Q21 = blocks_d[block_idx[0],block_idx[1]+1]
#     qt_mesh.Q22 = blocks_d[block_idx[0]+1,block_idx[1]+1]
#     verts, uvs, faces = qt_mesh.create_mesh(max_v, max_uv)
#     # if len(verts)==4:
#     # breakpoint()
#     if len(verts) > 0:
#         verts[:,:2] = vtx_ancor[:2] + verts[:,:2]
#         uvs = np.clip(uvs + uv_ancor, 0.0, 1.0)
#         faces = faces + vtx_idx
#     corners_idx = qt_mesh.mtx_verts[qt_mesh.corners][:,1]
#     return verts, uvs, faces, corners_idx

# def gen_edge_foreground_mesh(depth, canny, div_factor, verts, uvs, faces, stride):
#     # kernel = np.ones((3, 3), 'uint8')
#     # dilated_d = cv2.dilate(depth, kernel, iterations=1)
#     # depth[canny==255] = dilated_d[canny==255]
#     n_faces = faces      # Selected vertex faces there will be changed
#     n_vertices = verts     # Initial vertices
#     n_tex_uv = uvs
#     count = 0
#     max_v = verts[0].numpy() - verts[1].numpy()
#     max_uv = np.abs(uvs[0].numpy() - uvs[1].numpy())
#     max_uv = np.abs(max_uv.max())
#     max_v = max_v.max()
#     blocks_d = -1*np.ones((depth.shape[0]//div_factor+1, depth.shape[1]//div_factor+1))
#     qt_mesh_list = []
#     vtx_ancors = []
#     uv_ancors = []
#     idx_list = []
#     corners_v_idx = []
#     for i in range(0,depth.shape[0]-1,div_factor):
#         i_idx = i//div_factor
#         for j in range(0,depth.shape[1]-1,div_factor):
#             j_idx = j//div_factor
#             idx = i_idx*stride + j_idx
#             vtx_ancor = verts[idx].numpy()
#             uv_ancor = uvs[idx].numpy()
#             # breakpoint()
#             region_canny = canny[i:i+div_factor+1,j:j+div_factor+1]//255
#             if region_canny.shape[0] == (div_factor+1) and region_canny.shape[1] == (div_factor + 1):
#                 if (region_canny).sum() > 0:
#                     qt_mesh = gen_region_mesh(depth,
#                                             canny,
#                                             (i,i+div_factor),
#                                             (j,j+div_factor),
#                                             blocks_d,
#                                             (i_idx, j_idx))
#                     qt_mesh_list.append(qt_mesh)
#                     vtx_ancors.append(vtx_ancor)
#                     uv_ancors.append(uv_ancor)
#                     idx_list.append((i_idx, j_idx))
#                     corners_v_idx.append(np.array([idx, idx+1, idx + stride, idx+stride+1]))
#         #             count += 1
#         #             if count >1:
#         #                 break
#         # if count >1:
#         #     break

#     for i, qt_mesh in enumerate(qt_mesh_list):
#         out_verts, out_uvs, out_faces, corners_idx = gen_mesh_faces(qt_mesh,
#                                                        max_v,
#                                                        max_uv,
#                                                         vtx_ancors[i],
#                                                         uv_ancors[i],
#                                                         n_vertices.shape[0],
#                                                         blocks_d,
#                                                         idx_list[i])

#         if len(out_verts)>0:
#             mask = corners_idx>0
#             # breakpoint()
#             out_verts = torch.from_numpy(out_verts).to(torch.float32)
#             out_uvs = torch.from_numpy(out_uvs).to(torch.float32)
#             out_faces = torch.from_numpy(out_faces)
#             n_vertices[corners_v_idx[i][mask]] = out_verts[corners_idx[mask].astype(int)]
#             n_vertices = torch.cat([n_vertices,out_verts], dim=0)
#             n_tex_uv = torch.cat([n_tex_uv, out_uvs], dim=0)
#             n_faces = torch.cat([n_faces, out_faces], dim=0)


#         # if count > 0 :
#         #     break
#     n_vertices[:,2] = 1.0 - n_vertices[:,2]
#     return n_vertices, n_tex_uv, n_faces

# def get_background_faces(block_verts, block_uvs, idxs, V_MAX_IDX):
#     mtx_labels = np.zeros(4)
#     corners_depth = block_verts[:,-1]
#     max_depth_idx = corners_depth.argmax()
#     corners_diff = np.abs(corners_depth - corners_depth[max_depth_idx])
#     fore_corners = corners_diff < 1.1176e-1
#     back_corners = corners_diff >= 1.1176e-1
#     # breakpoint()
#     mtx_labels[back_corners] = 0
#     mtx_labels[fore_corners] = 1
#     oposes = np.array([3,2,1,0])
#     edges = np.array([[1,2],[0,3],[0,3],[1,2]])
#     n_faces = []
#     n_verts = []
#     n_uvs = []
#     if mtx_labels.sum() == 1:
#         l_vtx = np.where(mtx_labels==1)[0]
#         v0 = block_verts[mtx_labels==1][0]
#         v0[-1] = (block_verts[mtx_labels==0][:,-1].sum())/3

#         c_aux0 = block_uvs[3] - block_uvs[2]
#         c_aux1 = block_uvs[0] - block_uvs[2]
#         c_vet = c_aux0/2 + c_aux1/2
#         c_vert = block_uvs[2] + c_vet
#         n_verts.append(v0)
#         n_uvs.append(c_vert)
#         # if l_vtx[0]>len(idxs) or l_vtx[0]>len(edges):
#         n_faces.append([V_MAX_IDX, idxs[l_vtx][0],idxs[edges[l_vtx][0,0]]])
#         n_faces.append([V_MAX_IDX, idxs[l_vtx][0],idxs[edges[l_vtx][0,1]]])
#         n_faces.append([V_MAX_IDX, idxs[oposes[l_vtx]][0],idxs[edges[l_vtx][0,0]]])
#         n_faces.append([V_MAX_IDX, idxs[oposes[l_vtx]][0],idxs[edges[l_vtx][0,1]]])

#     elif mtx_labels.sum() == 3:
#         l_vtx = np.where(mtx_labels==0)[0]
#         c_aux0 = block_verts[1] - block_verts[0]
#         c_aux1 = block_verts[2] - block_verts[0]
#         c_vet = c_aux0/2 + c_aux1/2
#         c_vert = block_verts[0] + c_vet
#         v0 = c_vert
#         # breakpoint()
#         v0[-1] = block_verts[l_vtx][0,-1]

#         c_aux0 = block_uvs[3] - block_uvs[2]
#         c_aux1 = block_uvs[0] - block_uvs[2]
#         c_vet = c_aux0/2 + c_aux1/2
#         uv0 = block_uvs[2] + c_vet
#         n_verts.append(v0)
#         n_uvs.append(uv0)
#         n_faces.append([V_MAX_IDX, idxs[0], idxs[1]])
#         n_faces.append([V_MAX_IDX, idxs[0], idxs[2]])
#         n_faces.append([V_MAX_IDX, idxs[2], idxs[3]])
#         n_faces.append([V_MAX_IDX, idxs[3], idxs[1]])

#     elif mtx_labels.sum() == 2:
#         b_vtxs = np.where(mtx_labels==0)[0]
#         c_aux0 = block_verts[1] - block_verts[0]
#         c_aux1 = block_verts[2] - block_verts[0]
#         c_vet = c_aux0/2 + c_aux1/2
#         c_vert = block_verts[0] + c_vet
#         v0 = c_vert
#         v0[-1] = block_verts[b_vtxs][:,-1].sum()/2 # V_MAX_IDX

#         c_aux0 = block_uvs[3] - block_uvs[2]
#         c_aux1 = block_uvs[0] - block_uvs[2]
#         c_vet = c_aux0/2 + c_aux1/2
#         uv0 = block_uvs[2] + c_vet
#         n_verts.append(v0)
#         n_uvs.append(uv0)

#         count = 1
#         parallel_labels = False
#         for b_vt in b_vtxs:
#             vt_edges = edges[b_vt]
#             l_edges = mtx_labels[vt_edges]
#             for l_e, i_vt in zip(l_edges,vt_edges):
#                 if l_e==1:
#                     c_vet = block_verts[i_vt] - block_verts[b_vt]
#                     c_v = block_verts[b_vt] + c_vet
#                     c_v[-1] = block_verts[b_vt][-1]

#                     c_uvet = block_uvs[i_vt] - block_uvs[b_vt]
#                     c_uv = block_uvs[b_vt] + c_uvet/2
#                     n_verts.append(c_v)
#                     n_uvs.append(c_uv)
#                     # breakpoint()
#                     n_faces.append([V_MAX_IDX, V_MAX_IDX+count, idxs[b_vt]])
#                     n_faces.append([V_MAX_IDX, V_MAX_IDX+count, idxs[i_vt]])
#                     count +=1
#                 else:
#                     if not parallel_labels:
#                         n_faces.append([V_MAX_IDX, idxs[b_vt], idxs[i_vt]])
#                         n_faces.append([V_MAX_IDX, idxs[oposes[b_vt]], idxs[oposes[i_vt]]])
#                         parallel_labels = True

#     return np.array(n_verts), np.array(n_uvs), np.array(n_faces)

# def gen_edge_background_mesh(depth, canny, div_factor, verts, uvs, faces, stride):
#     n_faces = faces      # Selected vertex faces there will be changed
#     n_vertices = verts     # Initial vertices
#     n_tex_uv = uvs
#     b_vs = np.zeros((4,3))
#     b_uvs = np.zeros((4,2))
#     for i in range(0,depth.shape[0]-div_factor-1,div_factor):
#         i_idx = i//div_factor
#         for j in range(0,depth.shape[1]-div_factor-1,div_factor):
#             j_idx = j//div_factor
#             idx = i_idx*stride + j_idx
#             b_vs[0] = verts[idx].numpy()
#             b_vs[1] = verts[idx+1].numpy()
#             b_vs[2] = verts[idx+stride].numpy()
#             b_vs[3] = verts[idx+stride+1].numpy()
#             b_uvs[0] = uvs[idx].numpy()
#             b_uvs[1] = uvs[idx+1].numpy()
#             b_uvs[2] = uvs[idx+stride].numpy()
#             b_uvs[3] = uvs[idx+stride+1].numpy()
#             idxs = np.array([idx, idx+1, idx+stride, idx+stride+1])
#             # breakpoint()
#             region_canny = canny[i:i+div_factor+1,j:j+div_factor+1]//255
#             if region_canny.shape[0] == (div_factor+1) and region_canny.shape[1] == (div_factor + 1):
#                 if (region_canny).sum() > 0:
#                     out_v, out_uv, out_faces = get_background_faces(b_vs,b_uvs, idxs, n_vertices.shape[0])
#                     if len(out_v)>0:
#                         # breakpoint()
#                         out_verts = torch.from_numpy(out_v).to(torch.float32)
#                         out_uvs = torch.from_numpy(out_uv).to(torch.float32)
#                         out_faces = torch.from_numpy(out_faces)
#                         n_vertices = torch.cat([n_vertices,out_verts], dim=0)
#                         n_tex_uv = torch.cat([n_tex_uv, out_uvs], dim=0)
#                         n_faces = torch.cat([n_faces, out_faces], dim=0)

#     return n_vertices, n_tex_uv, n_faces


# def gen_not_edge_faces(depth, canny, div_factor, verts , stride, mask_img = None, vert_labels=None, img_index=None, img_vert=None):
#     n_faces = []      # Selected vertex faces there will be changed
#     n_vertices = verts
#     for i in range(0,depth.shape[0]-1,div_factor):
#         i_idx = i//div_factor
#         for j in range(0,depth.shape[1]-1,div_factor):
#             j_idx = j//div_factor
#             idx = i_idx*stride + j_idx
#             region_canny = canny[i:i+div_factor+1,j:j+div_factor+1]//255
#             region_mask = mask_img[i+1:i+div_factor,j+1:j+div_factor] if not mask_img is None else np.zeros(2)
#             if region_canny.shape[0] == (div_factor+1) and region_canny.shape[1] == (div_factor + 1):
#                 idx0 = idx
#                 idx1 = idx + 1
#                 idx2 = idx + stride
#                 idx3 = idx + stride + 1
#                 if (region_canny).sum() == 0:
#                     # if vert_labels[idx0]==1:
#                     img_index[i,j] = idx0
#                     img_vert[i,j,0]= 1
#                     img_vert[i,j,1]= depth[i,j]
#                     # if vert_labels[idx1]==1:
#                     img_index[i,j+div_factor] = idx1
#                     img_vert[i,j+div_factor,0]= 1
#                     img_vert[i,j+div_factor,1]= depth[i,j+div_factor]
#                     # if vert_labels[idx2]==1:
#                     img_index[i+div_factor,j] = idx2
#                     img_vert[i+div_factor,j,0]= 1
#                     img_vert[i+div_factor,j,1]= depth[i+div_factor,j]
#                     # if vert_labels[idx3]==1:
#                     img_index[i+div_factor,j+div_factor] = idx3
#                     img_vert[i+div_factor,j+div_factor,0]= 1
#                     img_vert[i+div_factor,j+div_factor,1]= depth[i+div_factor,j+div_factor]
#                     if region_mask.sum()==0:
#                         n_faces.append([idx1, idx2, idx0])
#                         n_faces.append([idx2, idx1, idx3])


#     return torch.tensor(n_faces), img_index, img_vert


# def compute_vertex_normals(faces,verts):
#         """Computes the packed version of vertex normals from the packed verts
#         and faces. This assumes verts are shared between faces. The normal for
#         a vertex is computed as the sum of the normals of all the faces it is
#         part of weighed by the face areas.

#         Args:
#             refresh: Set to True to force recomputation of vertex normals.
#                 Default: False.
#         """
#         faces_packed = faces
#         verts_packed = verts
#         verts_normals = torch.zeros_like(verts_packed)
#         vertices_faces = verts_packed[faces_packed]
#         # verts_00 = torch.index_select(verts_packed,0, faces_packed[::2,0])
#         # verts_10 = torch.index_select(verts_packed,0, faces_packed[::2,1])
#         # verts_20 = torch.index_select(verts_packed,0, faces_packed[::2,2])

#         # verts_01 = torch.index_select(verts_packed,0, faces_packed[1::2,0])
#         # verts_11 = torch.index_select(verts_packed,0, faces_packed[1::2,1])
#         # verts_21 = torch.index_select(verts_packed,0, faces_packed[1::2,2])

#         faces_normals = torch.cross(
#             vertices_faces[:, 2] - vertices_faces[:, 1],
#             vertices_faces[:, 0] - vertices_faces[:, 1],
#             dim=1,
#         )

#         # NOTE: this is already applying the area weighting as the magnitude
#         # of the cross product is 2 x area of the triangle.
#         verts_normals = verts_normals.index_add(
#             0, faces_packed[:, 0], faces_normals
#         )
#         verts_normals = verts_normals.index_add(
#             0, faces_packed[:, 1], faces_normals
#         )
#         verts_normals = verts_normals.index_add(
#             0, faces_packed[:, 2], faces_normals
#         )
#         # breakpoint()
#         verts_normals_packed = torch.nn.functional.normalize(
#             verts_normals, eps=1e-6, dim=1
#         )

#         return verts_normals_packed

# def compute_edges(faces):
#     F = faces.shape[0]
#     v0, v1, v2 = faces.chunk(3, dim=1)
#     e01 = torch.cat([v0, v1], dim=1)  # (sum(F_n), 2)
#     e12 = torch.cat([v1, v2], dim=1)  # (sum(F_n), 2)
#     e20 = torch.cat([v2, v0], dim=1)  # (sum(F_n), 2)

#     # All edges including duplicates.
#     edges = torch.cat([e12, e20, e01], dim=0)  # (sum(F_n)*3, 2)
#     return edges, e01, e12, e20


# def compute_edge_length(offset_vertices, edges_packed):

#     # edges_packed = compute_edges(faces)  # (sum(E_n), 2)
#     verts_packed = offset_vertices  # (sum(V_n), 3)

#     # edges_packed = edges_packed[None, ...]
#     # edges_packed = edges_packed.expand(offset_vertices.shape[0], *edges_packed.shape[1:])
#     verts_edges = verts_packed[edges_packed.to(torch.int64)]
#     # breakpoint()
#     v0, v1 = verts_edges.unbind(1)
#     lengths = (v0 - v1).norm(dim=1, p=2)

#     return lengths

# def general_parallax_map(uv, depth, view_dir):
#     uv_np = uv.cpu().detach().numpy()
#     pos = np.clip(uv_np*(depth.shape[0]-1), 0, depth.shape[0]).astype(np.int32)
#     pos = pos.reshape(depth.shape[0], depth.shape[1], 2)
#     uv_depth = np.ones_like(pos).astype(np.float32)
#     height_scale = 0.5
#     for i in range(pos.shape[0]):
#         for j in range(pos.shape[1]):
#             height = depth[pos[i,j,0], pos[i,j,1]]
#             p = (view_dir[:-1]/view_dir[-1]) * (height) * height_scale
#             # p[1] = -p[1]
#             uv_depth[i,j,:] = uv_np[i*depth.shape[0] + j,:] - p

#     return torch.from_numpy(uv_depth.reshape(-1,2))

# def parallax_map(uv, depth, view_dir):
#     uv_np = uv.cpu().detach().numpy()
#     pos = np.clip(uv_np*(depth.shape[0]-1), 0, depth.shape[0]).astype(np.int32)
#     pos = pos.reshape(depth.shape[0], depth.shape[1], 2)
#     uv_depth = np.ones_like(pos).astype(np.float32)
#     height_scale = 0.1
#     for i in range(pos.shape[0]):
#         for j in range(pos.shape[1]):
#             height = depth[pos[i,j,0], pos[i,j,1]]
#             p = (-1.0)*(view_dir[:-1]/view_dir[-1]) * (1.0-height) * height_scale

#             uv_depth[i,j,:] = uv_np[i*depth.shape[0] + j,:] - p

#     return torch.from_numpy(uv_depth.reshape(-1,2))

# def get_depth_from_vert(vert, depth):
#     i_h = torch.round((((1.0-vert[1]))/2.0)*depth.shape[1]).to(torch.int64)
#     i_w = torch.round((((1.0-vert[0]))/2.0)*depth.shape[0]).to(torch.int64)
#     if i_h >= depth.shape[0]:
#         i_h = depth.shape[0] - 1

#     if i_w >= depth.shape[1]:
#         i_w = depth.shape[1] - 1
#     return 1.0 - depth[i_h, i_w, 0]

# def divide_face(face, verts, tex_uv, orig_depth, index, edge_th = 0.001):
#     new_verts = []
#     face_np = face.cpu().detach().numpy()
#     v = verts[face]
#     uv = tex_uv[face]

#     dir_0i0 = (v[1] - v[0])
#     dir_0i1 = (v[2] - v[1])
#     dir_0i2 = (v[0] - v[2])
#     new_verts = []
#     new_uvs = []
#     if dir_0i0 >= edge_th:
#         v = 0.5*dir_0i0 + v[0]
#         d = get_depth_from_vert(v, orig_depth)
#         v[-1] = d
#         new_verts.append(v)
#         new_uvs.append(0.5*(uv[1] - uv[0]) + uv[0])
#     if dir_0i1 >= edge_th:
#         v = 0.5*dir_0i1 + v[1]
#         d = get_depth_from_vert(v, orig_depth)
#         v[-1] = d
#         new_verts.append(v)
#         new_uvs.append(0.5*(uv[2] - uv[1]) + uv[1])
#     if dir_0i2 > edge_th:
#         v = 0.5*dir_0i2 + v[2]
#         d = get_depth_from_vert(v, orig_depth)
#         v[-1] = d
#         new_verts.append(v)
#         new_uvs.append(0.5*(uv[0] - uv[2]) + uv[2])

#     # v_0i0 = 0.5*(v[1] - v[0]) + v[0]
#     # v_0i1 = 0.5*(v[2] - v[1]) + v[1]
#     # v_0i2 = 0.5*(v[0] - v[2]) + v[2]

#     # uv_0i0 = 0.5*(uv[1] - uv[0]) + uv[0]
#     # uv_0i1 = 0.5*(uv[2] - uv[1]) + uv[1]
#     # uv_0i2 = 0.5*(uv[0] - uv[2]) + uv[2]

#     # a = get_depth_from_vert(v_0i0, orig_depth)
#     # v_0i0[-1] = a
#     # v_0i1[-1] = get_depth_from_vert(v_0i1, orig_depth)
#     # v_0i2[-1] = get_depth_from_vert(v_0i2, orig_depth)

#     o_i = index
#     new_faces = []

#     if len(new_verts) ==3:
#         face_0 = torch.tensor([face_np[0], o_i, o_i+2])
#         face_1 = torch.tensor([face_np[1], o_i+1, o_i])
#         face_2 = torch.tensor([face_np[2], o_i+2, o_i+1])
#         face_3 = torch.tensor([o_i+2, o_i, o_i+1])

#     out_faces = [face_0, face_1, face_2, face_3]
#     out_verts = [v_0i0, v_0i1, v_0i2]
#     out_uvs = [uv_0i0, uv_0i1, uv_0i2]

#     return out_faces, out_verts, out_uvs

# def split_faces(faces, verts, tex_uv, orig_depth):

#     n_faces = []
#     n_vertices = []
#     n_tex_uv = []
#     idx = verts.shape[0]
#     for f in faces:
#         out_faces, out_verts, out_uvs = divide_face(f, verts, tex_uv, orig_depth, idx)
#         n_faces += out_faces
#         n_vertices += out_verts
#         n_tex_uv += out_uvs
#         idx += 3
#     n_faces = torch.stack(n_faces)
#     n_vertices = torch.stack(n_vertices)
#     n_tex_uv = torch.stack(n_tex_uv)
#     return n_faces, n_vertices, n_tex_uv

# def mark_edges(edges, verts):
#     edges_len = compute_edge_length(verts, edges)
#     M_edges = torch.where(edges_len>(edges_len.mean() + 0.5*edges_len.mean()), True, False)

#     return edges_len, M_edges

# def get_local_verts(edges, f_edges, f_vert, verts, face_label, edges_len):
#     f_edge_len = edges_len[f_edges]
#     f_edge_idx = edges[f_edges]

#     if face_label == 1:
#         idx_max_len = f_edge_len.argmax()
#         max_edge = f_edge_idx[idx_max_len]
#         f_vert_aux = f_vert[f_vert!=max_edge[0]]
#         v_c = f_vert_aux[f_vert_aux!=max_edge[1]][0]
#         v_a = max_edge[0]
#         v_b = max_edge[1]

#         rem_mask = torch.where(torch.arange(3)!=idx_max_len, True, False)
#         f_edge_out = f_edges[rem_mask]
#         # breakpoint()

#     elif face_label == 2:
#         idx_max_len0 = f_edge_len.argmax()
#         edge_max_len0 = f_edge_len[idx_max_len0]
#         max_edge_0 = f_edge_idx[idx_max_len0]

#         rem_mask = torch.where(torch.arange(3)!=idx_max_len0, True, False)
#         f_edge_len = f_edge_len[rem_mask]
#         f_edge_idx = f_edge_idx[rem_mask]
#         f_edges_rm = f_edges[rem_mask]

#         idx_max_len1 = f_edge_len.argmax()
#         edge_max_len1 = f_edge_len[idx_max_len1]
#         max_edge_1 = f_edge_idx[idx_max_len1]
#         rem_mask = torch.where(torch.arange(2)!=idx_max_len1, True, False)
#         f_edge_out = f_edges_rm[rem_mask]

#         v_b = np.intersect1d(max_edge_0, max_edge_1)[0]
#         v_a = max_edge_0[max_edge_0!=v_b][0]
#         v_c = max_edge_1[max_edge_1!=v_b][0]

#     elif face_label == 3:
#         v_a = f_vert[0]
#         v_b = f_vert[1]
#         v_c = f_vert[2]
#         f_edge_out = None

#     return v_a, v_b, v_c, f_edge_out


# def check_labels(f_edge, M_edges):
#     labels = M_edges[f_edge]
#     return labels.sum()

# def get_intersect_point(v_a0, v_c0, v_bxc, v_axb):
#     zs = torch.zeros(4)
#     if len(v_a0)==3:
#         x1, y1, zs[0] = v_a0[0], v_a0[1], v_a0[2]
#         x2, y2, zs[1] = v_bxc[0], v_bxc[1], v_bxc[2]
#         x3, y3, zs[2] = v_c0[0], v_c0[1], v_c0[2]
#         x4, y4, zs[3] = v_axb[0], v_axb[1],v_axb[2]
#     else:
#         x1, y1 = v_a0[0], v_a0[1]
#         x2, y2 = v_bxc[0], v_bxc[1]
#         x3, y3 = v_c0[0], v_c0[1]
#         x4, y4 = v_axb[0], v_axb[1]

#     t = (x1 - x3)*(y3 - y4) - (y1 - y3)*(x3 - x4)
#     t = t / ((x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4))

#     P_x, P_y  = x1 + t*(x2 - x1), y1 + t*(y2 - y1)
#     P_z = zs.mean()
#     return P_x, P_y, P_z

# # edges_struct:
# #   - faces_con

# def gen_new_faces(edges, f_edge, f_vert, verts, uvs, face_label, edges_len, origig_depth, V_END, E_END):
#     va, vb, vc, f_edge_out = get_local_verts(edges, f_edge, f_vert, verts, face_label, edges_len)
#     new_verts = []
#     new_uv = []
#     new_faces = []
#     new_edges = []
#     new_F_edges = []
#     # breakpoint()
#     if face_label == 0:
#         return new_verts, new_uv, new_faces

#     elif face_label == 1:
#         v_a0 = verts[va]
#         v_b0 = verts[vb]
#         uv_a0 = uvs[va]
#         uv_b0 = uvs[vb]

#         v_axb = 0.5*(v_b0 - v_a0) + v_a0
#         v_axb[-1] = get_depth_from_vert(v_axb, origig_depth)
#         uv_axb = 0.5*(uv_b0 - uv_a0) + uv_a0
#         v_axb_idx = V_END

#         f0 = torch.tensor([va, v_axb_idx, vc])
#         f1 = torch.tensor([vc, v_axb_idx, vb])

#         e0 = torch.tensor([va, v_axb_idx]) # E_END
#         e1 = torch.tensor([v_axb_idx, vc]) # E_END + 1
#         e2 = torch.tensor([v_axb_idx, vb]) # E_END + 2

#         f_e0 = torch.tensor([f_edge_out[0], E_END, E_END+1])
#         f_e1 = torch.tensor([E_END+1, E_END+2, f_edge_out[1]])

#         new_verts = [v_axb]
#         new_uv = [uv_axb]
#         new_faces = [f0, f1]
#         new_edges = [e0,e1,e2]
#         new_F_edges = [f_e0, f_e1]

#     elif face_label == 2:
#         v_a0 = verts[va]
#         v_b0 = verts[vb]
#         v_c0 = verts[vc]
#         uv_a0 = uvs[va]
#         uv_b0 = uvs[vb]
#         uv_c0 = uvs[vc]
#         v_axb = 0.5*(v_b0 - v_a0) + v_a0
#         v_axb[-1] = get_depth_from_vert(v_axb, origig_depth)
#         v_bxc = 0.5*(v_c0 - v_b0) + v_b0
#         v_bxc[-1] = get_depth_from_vert(v_bxc, origig_depth)
#         uv_axb = 0.5*(uv_b0 - uv_a0) + uv_a0
#         uv_bxc = 0.5*(uv_c0 - uv_b0) + uv_b0
#         v_axb_idx = V_END
#         v_bxc_idx = V_END + 1
#         v_abc_idx = V_END + 2
#         v_abc = get_intersect_point(v_a0, v_c0, v_bxc, v_axb)
#         uv_abc = get_intersect_point(uv_a0, uv_c0, uv_bxc, uv_axb)[:2]
#         v_abc = torch.tensor(v_abc)
#         v_abc[-1] = get_depth_from_vert(v_abc, origig_depth)
#         uv_abc = torch.tensor(uv_abc)
#         # uv_abc = (v_abc[:2]+1.0)/2.0
#         # uv_abc[0] = (1.0 - uv_abc[0])
#         # breakpoint()

#         f0 = torch.tensor([va, v_abc_idx, vc])
#         f1 = torch.tensor([va, v_abc_idx, v_axb_idx])
#         f2 = torch.tensor([v_axb_idx, v_abc_idx, vb])
#         f3 = torch.tensor([vb, v_abc_idx, v_bxc_idx])
#         f4 = torch.tensor([v_bxc_idx, v_abc_idx, vc])

#         e0 = torch.tensor([v_abc_idx, vc]) # E_END
#         e1 = torch.tensor([va, v_abc_idx]) # E_END + 1
#         e2 = torch.tensor([va, v_axb_idx]) # E_END + 2
#         e3 = torch.tensor([v_abc_idx, v_axb_idx]) # E_END + 3
#         e4 = torch.tensor([v_axb_idx, vb]) # E_END + 4
#         e5 = torch.tensor([v_abc_idx, vb]) # E_END + 5
#         e6 = torch.tensor([v_bxc_idx, vb]) # E_END + 6
#         e7 = torch.tensor([v_abc_idx, v_bxc_idx]) # E_END + 7
#         e8 = torch.tensor([vc, v_bxc_idx]) # E_END + 8

#         f_e0 = torch.tensor([f_edge_out[0], E_END, E_END+1])
#         f_e1 = torch.tensor([E_END+1, E_END+2, E_END+3])
#         f_e2 = torch.tensor([E_END+3, E_END+4, E_END+5])
#         f_e3 = torch.tensor([E_END+5, E_END+6, E_END+7])
#         f_e4 = torch.tensor([E_END+7, E_END+8, E_END])

#         new_verts = [v_axb, v_bxc, v_abc]
#         new_uv = [uv_axb, uv_bxc, uv_abc]
#         new_faces = [f0, f1, f2, f3, f4]
#         new_edges = [e0, e1, e2, e3, e4, e5, e6, e7, e8]
#         new_F_edges = [f_e0, f_e1, f_e2, f_e3, f_e4]

#     else:
#         v_a0 = verts[va]
#         v_b0 = verts[vb]
#         v_c0 = verts[vc]
#         uv_a0 = uvs[va]
#         uv_b0 = uvs[vb]
#         uv_c0 = uvs[vc]
#         v_axb = 0.5*(v_b0 - v_a0) + v_a0
#         v_axb[-1] = get_depth_from_vert(v_axb, origig_depth)
#         v_bxc = 0.5*(v_c0 - v_b0) + v_b0
#         v_bxc[-1] = get_depth_from_vert(v_bxc, origig_depth)
#         v_cxa = 0.5*(v_a0 - v_c0) + v_c0
#         v_cxa[-1] = get_depth_from_vert(v_cxa, origig_depth)
#         uv_axb = 0.5*(uv_b0 - uv_a0) + uv_a0
#         uv_bxc = 0.5*(uv_c0 - uv_b0) + uv_b0
#         uv_cxa = 0.5*(uv_a0 - uv_c0) + uv_c0
#         v_axb_idx = V_END
#         v_bxc_idx = V_END + 1
#         v_cxa_idx = V_END + 2

#         f0 = torch.tensor([va, v_axb_idx, v_cxa_idx])
#         f1 = torch.tensor([v_axb_idx, vb, v_bxc_idx])
#         f2 = torch.tensor([v_bxc_idx, vc, v_cxa_idx])
#         f3 = torch.tensor([v_cxa_idx, v_axb_idx, v_bxc_idx])

#         e0 = torch.tensor([v_cxa_idx, va]) # E_END
#         e1 = torch.tensor([va, v_axb_idx]) # E_END + 1
#         e2 = torch.tensor([v_axb_idx, v_cxa_idx]) # E_END + 2
#         e3 = torch.tensor([v_axb_idx, vb]) # E_END + 3
#         e4 = torch.tensor([vb, v_bxc_idx]) # E_END + 5
#         e5 = torch.tensor([v_bxc_idx, v_axb_idx]) # E_END + 6
#         e6 = torch.tensor([v_bxc_idx, v_cxa_idx]) # E_END + 7
#         e7 = torch.tensor([vc, v_bxc_idx]) # E_END + 8
#         e8 = torch.tensor([vc, v_cxa_idx]) # E_END + 8


#         f_e0 = torch.tensor([E_END, E_END+1, E_END+2])
#         f_e1 = torch.tensor([E_END+3, E_END+4, E_END+5])
#         f_e2 = torch.tensor([E_END+6, E_END+7, E_END+8])
#         f_e3 = torch.tensor([E_END+2, E_END+5, E_END+6])

#         new_verts = [v_axb, v_bxc, v_cxa]
#         new_uv = [uv_axb, uv_bxc, uv_cxa]
#         new_faces = [f0, f1, f2, f3]
#         new_edges = [e0, e1, e2, e3, e4, e5, e6, e7, e8]
#         new_F_edges = [f_e0, f_e1, f_e2, f_e3]


#     return new_verts, new_uv, new_faces, new_edges, new_F_edges

# def create_faces(edges, M_edges, S_faces, SE_faces, NS_faces, edges_len, verts, uvs, orig_depth, degree = 4):

#     n_faces = [NS_faces]      # Selected vertex faces there will be changed
#     n_vertices = verts     # Initial vertices
#     n_tex_uv = uvs         # Initial uvs
#     n_edges = edges        # Initial edges
#     n_F_edges = SE_faces   # Selected edge faces
#     it_edges_len = edges_len
#     idx = verts.shape[0]
#     idx_e = edges.shape[0]

#     # Arrays that will be used to iterate into faces to generate new faces
#     # In each iteration, these arrays receave the selected new faces
#     it_SE_faces, it_S_faces = SE_faces, S_faces
#     it_M_edges = M_edges
#     for it in range(degree):
#         it_faces = []
#         it_vertices = []
#         it_tex_uv = []
#         it_edges = []
#         it_F_edges = []

#         # Generate new faces
#         for (f_edge, f_vert) in zip(it_SE_faces, it_S_faces):
#             face_label = check_labels(f_edge, it_M_edges)
#             out_verts, out_uvs, out_faces, out_edges, out_F_edges = gen_new_faces(n_edges, f_edge, f_vert, n_vertices, n_tex_uv, face_label, it_edges_len, orig_depth, idx, idx_e)
#             it_faces += out_faces
#             it_vertices += out_verts
#             it_tex_uv += out_uvs
#             it_edges += out_edges
#             it_F_edges += out_F_edges
#             idx += len(out_verts)
#             idx_e += len(out_edges)

#         it_faces = torch.stack(it_faces)
#         it_vertices = torch.stack(it_vertices)
#         it_tex_uv = torch.stack(it_tex_uv)
#         it_edges = torch.stack(it_edges)
#         it_F_edges = torch.stack(it_F_edges)

#         # n_faces = torch.cat([n_faces,it_faces], dim=0)
#         n_vertices = torch.cat([n_vertices,it_vertices], dim=0)
#         n_edges = torch.cat([n_edges,it_edges], dim=0)
#         n_tex_uv = torch.cat([n_tex_uv,it_tex_uv], dim=0)
#         it_edges_len, it_M_edges = mark_edges(n_edges, n_vertices)

#         if it < (degree - 1):
#             m_s_faces = it_M_edges[it_F_edges]
#             m_s_faces = m_s_faces.sum(dim=1)
#             s_faces = torch.where(m_s_faces>0, True, False)
#             ns_faces = torch.where(m_s_faces>0, False, True)
#             it_S_faces = it_faces[s_faces]
#             it_NS_faces = it_faces[ns_faces]
#             it_SE_faces = it_F_edges[s_faces]
#             n_faces += [it_NS_faces]
#         else:
#             n_faces = torch.cat(n_faces,dim=0)
#             n_faces = torch.cat([n_faces, it_faces], dim=0)



#     return n_faces, n_vertices, n_tex_uv


# def get_back_fore(verts, edges, M_edges):
#     bg_verts = torch.zeros(verts.shape[0]).to(torch.int32)
#     fg_verts = torch.zeros(verts.shape[0]).to(torch.int32)
#     final_labels = torch.zeros(verts.shape[0]).to(torch.int32)
#     s_edges = edges[M_edges]
#     aux = torch.ones(s_edges.shape[0]).to(torch.int32)
#     # v0, v1 = verts[s_edges].unbind(1)
#     # aux = (v0[:,-1] > v1[:,-1]).to(torch.int32)
#     # neg_aux = 1 - aux
#     # bg_verts = bg_verts.index_add(
#     #     0, s_edges[:,0], aux
#     # )

#     fg_verts = fg_verts.index_add(
#         0, s_edges[:,0], aux
#     )
#     # aux = (v0[:,-1] < v1[:,-1]).to(torch.int32)
#     # bg_verts = bg_verts.index_add(
#     #     0, s_edges[:,1], neg_aux
#     # )

#     fg_verts = fg_verts.index_add(
#         0, s_edges[:,1], aux
#     )

#     # final_labels[bg_verts>fg_verts] = 1
#     mask = fg_verts>0
#     mask = mask + fg_verts<5
#     final_labels[mask] = 2
#     return final_labels


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

        v_pos = v_pos.detach().cpu().numpy() if v_pos is not None else None
        v_nrm = v_nrm.detach().cpu().numpy() if v_nrm is not None else None
        v_tex = v_tex.detach().cpu().numpy() if v_tex is not None else None

        t_pos_idx = t_pos_idx.detach().cpu().numpy() if t_pos_idx is not None else None
        t_nrm_idx = t_nrm_idx.detach().cpu().numpy() if t_nrm_idx is not None else None
        t_tex_idx = t_tex_idx.detach().cpu().numpy() if t_tex_idx is not None else None

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
