import halide as hl


@hl.alias(
    back_verts_Adams2019={"autoscheduler": "Adams2019"},
    back_verts_Mullapudi2016={"autoscheduler": "Mullapudi2016"},
    back_verts_Li2018={"autoscheduler": "Li2018"},
)

# Apply a I_labels value to a 2D image using a logical operator that is selected at compile-time.
@hl.generator(name = "back_verts")
class BackgroundVertsGenerator:
    block_size = hl.GeneratorParam(16)

    canny = hl.InputBuffer(hl.UInt(8), 2)
    back_canny = hl.InputBuffer(hl.UInt(8), 2)
    depth = hl.InputBuffer(hl.UInt(8), 2)
    block_stride_x = hl.InputScalar(hl.Int(32))
    block_stride_y = hl.InputScalar(hl.Int(32))


    # Variable with dimensions b_H x b_W
    BI_merge_mask = hl.OutputBuffer(hl.UInt(8), 2)
    BI_Vcoords = hl.OutputBuffer(hl.Float(32), 3)
    BI_fore_Vcoords = hl.OutputBuffer(hl.Float(32), 3)
    BI_UVcoords = hl.OutputBuffer(hl.Float(32), 3)

    I_labels = hl.OutputBuffer(hl.UInt(8), 2)

    # Depth image variables
    x, y, c, l, i, i_f = hl.vars("x y c l i i_f")

    # Block image variables
    b_x, b_y, b_li, b_c, b_cd = hl.vars("b_x b_y b_li b_c b_cd")

    def generate(G):
        # Algorithm
        W = G.canny.width()
        H = G.canny.height()
        canny_clamped = hl.BoundaryConditions.repeat_edge(G.canny)
        back_canny_clamped = hl.BoundaryConditions.repeat_edge(G.back_canny)
        depth_clamped = hl.BoundaryConditions.repeat_edge(G.depth, [(0,W),(0,H)])

        b_canny_r = hl.RDom([(0, G.block_size+1), (0, G.block_size+1)])
        canny_srch_exp = canny_clamped[(G.b_x * G.block_size) + b_canny_r.x, (G.b_y * G.block_size) + b_canny_r.y]
        block_n_canny_pixels = hl.Func("block_n_canny_pixels")
        block_n_canny_pixels[G.b_x, G.b_y] = hl.sum(
            hl.select(
                canny_srch_exp>0,1,0
            )
        )

        b_bcanny_r = hl.RDom([(0, G.block_size+1), (0, G.block_size+1)])
        back_canny_srch_exp = back_canny_clamped[(G.b_x * G.block_size) + b_bcanny_r.x, (G.b_y * G.block_size) + b_bcanny_r.y]
        block_n_bcanny_pixels = hl.Func("block_n_bcanny_pixels")
        block_n_bcanny_pixels[G.b_x, G.b_y] = hl.sum(
            hl.select(
                back_canny_srch_exp>0, 1, 0
            )
        )

        block_m_canny = hl.Func("block_m_canny")
        block_m_canny[G.b_x, G.b_y] = hl.select(
            block_n_canny_pixels[G.b_x, G.b_y]>0, 1, 0
        )

        b_depth_r = hl.RDom([(0, G.block_size+1), (0, G.block_size+1)])
        block_mean_depth = hl.Func("block_mean_depth")
        depth_srch_exp = depth_clamped[(G.b_x * G.block_size) + b_depth_r.x, (G.b_y * G.block_size) + b_depth_r.y]
        # block_mean_depth[G.b_x, G.b_y] = hl.sum(hl.f32(depth_srch_exp))/hl.f32((G.block_size+1)*(G.block_size+1))/255.0
        # block_mean_depth[G.b_x, G.b_y] = 255.0
        block_mean_depth[G.b_x, G.b_y] = hl.minimum(hl.f32(depth_srch_exp))/255.0


        srch_X = (G.b_x * G.block_size) + b_depth_r.x
        srch_Y = (G.b_y * G.block_size) + b_depth_r.y
        fore_mask = hl.Func("fore_mask")
        fore_mask[G.x,G.y] = hl.f32(depth_clamped[G.x, G.y])*(hl.f32(canny_clamped[G.x,G.y])/hl.f32(255))
        back_mask = hl.Func("back_mask")
        back_mask[G.x,G.y] = hl.f32(depth_clamped[G.x, G.y])*(hl.f32(back_canny_clamped[G.x,G.y])/hl.f32(255))

        block_fore_depth = hl.Func("block_fore_depth")
        sum_select_expr = hl.f32(hl.sum(fore_mask[srch_X,srch_Y]))
        select_n_expr = hl.f32(hl.select(block_n_canny_pixels[G.b_x, G.b_y]>0, block_n_canny_pixels[G.b_x, G.b_y], 1))
        block_fore_depth[G.b_x, G.b_y] = sum_select_expr/select_n_expr

        block_back_depth = hl.Func("block_back_depth")
        back_sum_select_expr = hl.f32(hl.sum(back_mask[srch_X,srch_Y]))
        back_select_n_expr = hl.f32(hl.select(block_n_bcanny_pixels[G.b_x, G.b_y]>0, block_n_bcanny_pixels[G.b_x, G.b_y], 1))
        block_back_depth[G.b_x, G.b_y] = back_sum_select_expr/back_select_n_expr

        corner1_fore_diff_expr = hl.abs(block_fore_depth[G.b_x, G.b_y] - depth_clamped[(G.b_x * G.block_size), (G.b_y * G.block_size)])
        corner2_fore_diff_expr = hl.abs(block_fore_depth[G.b_x, G.b_y] - depth_clamped[(G.b_x * G.block_size) + G.block_size, (G.b_y * G.block_size)])
        corner3_fore_diff_expr = hl.abs(block_fore_depth[G.b_x, G.b_y] - depth_clamped[(G.b_x * G.block_size), (G.b_y * G.block_size)+G.block_size])
        corner4_fore_diff_expr = hl.abs(block_fore_depth[G.b_x, G.b_y] - depth_clamped[(G.b_x * G.block_size) + G.block_size, (G.b_y * G.block_size) + G.block_size])

        corner1_back_diff_expr = hl.abs(block_back_depth[G.b_x, G.b_y] - depth_clamped[(G.b_x * G.block_size), (G.b_y * G.block_size)])
        corner2_back_diff_expr = hl.abs(block_back_depth[G.b_x, G.b_y] - depth_clamped[(G.b_x * G.block_size) + G.block_size, (G.b_y * G.block_size)])
        corner3_back_diff_expr = hl.abs(block_back_depth[G.b_x, G.b_y] - depth_clamped[(G.b_x * G.block_size), (G.b_y * G.block_size) + G.block_size])
        corner4_back_diff_expr = hl.abs(block_back_depth[G.b_x, G.b_y] - depth_clamped[(G.b_x * G.block_size) + G.block_size, (G.b_y * G.block_size) + G.block_size])

        block_canny_labels = hl.Func("block_canny_labels")
        block_canny_labels[G.b_x, G.b_y, G.b_c] = hl.u8(0)
        block_canny_labels[G.b_x, G.b_y, 0] = hl.u8(hl.select(corner1_back_diff_expr <= corner1_fore_diff_expr,
                                                              1, # background
                                                              2)) # foreground
        block_canny_labels[G.b_x, G.b_y, 1] = hl.u8(hl.select(corner2_back_diff_expr <= corner2_fore_diff_expr, 1, 2))
        block_canny_labels[G.b_x, G.b_y, 2] = hl.u8(hl.select(corner3_back_diff_expr <= corner3_fore_diff_expr, 1, 2))
        block_canny_labels[G.b_x, G.b_y, 3] = hl.u8(hl.select(corner4_back_diff_expr <= corner4_fore_diff_expr, 1, 2))

        x_block_check = (((G.x%G.block_size == 0) | ((G.y%G.block_size == 0) & (G.x==(W-1)))) | ((G.x==(W-1)) & (G.y==(H-1))))
        y_block_check = (((G.y%G.block_size == 0) | ((G.x%G.block_size == 0) & (G.y==(H-1)))) | ((G.x==(W-1)) & (G.y==(H-1))))
        x_b_expr = hl.i32(hl.floor(G.x/G.block_size))
        y_b_expr = hl.i32(hl.floor(G.y/G.block_size))

        count_labels_img = hl.Func("count_labels_img")
        count_labels_img[G.x, G.y, G.l] = 0
        count_labels_img[G.x, G.y, 0] = hl.select(
            x_block_check & y_block_check,
            (
             hl.select((block_canny_labels[x_b_expr, y_b_expr, 0] == 1)     & (block_m_canny[x_b_expr, y_b_expr]==1)  , 1, 0)
             + hl.select((block_canny_labels[x_b_expr-1, y_b_expr, 1]== 1)   & (block_m_canny[x_b_expr-1, y_b_expr]==1), 1, 0)
             + hl.select((block_canny_labels[x_b_expr, y_b_expr-1, 2]== 1)   & (block_m_canny[x_b_expr, y_b_expr-1]==1), 1, 0)
             + hl.select((block_canny_labels[x_b_expr-1, y_b_expr-1, 3]== 1) & (block_m_canny[x_b_expr-1, y_b_expr-1]==1)  , 1, 0)
             ),
             0
        )
        count_labels_img[G.x, G.y, 1] = hl.select(
            x_block_check & y_block_check,
            (
             hl.select((block_canny_labels[x_b_expr, y_b_expr, 0] == 2)       & (block_m_canny[x_b_expr, y_b_expr]==1)  , 1, 0)
             + hl.select((block_canny_labels[x_b_expr-1, y_b_expr, 1] == 2  ) & (block_m_canny[x_b_expr-1, y_b_expr]==1), 1, 0)
             + hl.select((block_canny_labels[x_b_expr, y_b_expr-1, 2] == 2  ) & (block_m_canny[x_b_expr, y_b_expr-1]==1), 1, 0)
             + hl.select((block_canny_labels[x_b_expr-1, y_b_expr-1, 3] == 2) & (block_m_canny[x_b_expr-1, y_b_expr-1]==1  ), 1, 0)
             ),
             0
        )

        first_labels_img = hl.Func("first_labels_img")
        comp_n_fore_back_l = count_labels_img[G.x,G.y,0] >= count_labels_img[G.x,G.y,1]
        first_labels_img[G.x,G.y] = hl.select(x_block_check & y_block_check, hl.select(comp_n_fore_back_l, 1, 2), 0)


        corner0_X = (G.b_x * G.block_size)
        corner1_X = (G.b_x * G.block_size) + G.block_size
        corner2_X = (G.b_x * G.block_size)
        corner3_X = (G.b_x * G.block_size) + G.block_size

        corner0_Y = (G.b_y * G.block_size)
        corner1_Y = (G.b_y * G.block_size)
        corner2_Y = (G.b_y * G.block_size) + G.block_size
        corner3_Y = (G.b_y * G.block_size) + G.block_size

        check_corner0 = first_labels_img[corner0_X, corner0_Y] == 2
        check_corner1 = first_labels_img[corner1_X, corner1_Y] == 2
        check_corner2 = first_labels_img[corner2_X, corner2_Y] == 2
        check_corner3 = first_labels_img[corner3_X, corner3_Y] == 2

        number_corners = (
            hl.mux(check_corner0, [1,2])
            * hl.mux(check_corner1, [1,3])
            * hl.mux(check_corner2, [1,5])
            * hl.mux(check_corner3, [1,7])
        )

        block_orientation = hl.Func("block_orientation")
        block_orientation[G.b_x, G.b_y] = hl.select(
            block_m_canny[G.b_x, G.b_y]==1, number_corners, 0
        )
        block_ori_cond = hl.BoundaryConditions.repeat_edge(block_orientation,[(0,hl.i32(G.canny.width()/G.block_size)),(0,hl.i32(G.canny.height()/G.block_size))])

        check_corners = check_corner0 | check_corner1 | check_corner2 | check_corner3
        not_b_canny = block_m_canny[G.b_x, G.b_y]==0

        #Neighbors idx configuration in block_idx_neighbors
        #------------
        # 0 | 1 | 2 |
        #------------
        # 3 |   | 4 |
        #------------
        # 5 | 6 | 7 |
        #------------

        neighbors = [
            block_ori_cond[G.b_x-1,G.b_y-1],
            block_ori_cond[G.b_x,G.b_y-1]  ,
            block_ori_cond[G.b_x+1,G.b_y-1],
            block_ori_cond[G.b_x-1,G.b_y]  ,
            block_ori_cond[G.b_x+1,G.b_y]  ,
            block_ori_cond[G.b_x-1,G.b_y+1],
            block_ori_cond[G.b_x,G.b_y+1]  ,
            block_ori_cond[G.b_x+1,G.b_y+1]
        ]

        block_ori_neighbors = hl.Func("block_ori_neighbors")
        block_ori_neighbors[G.b_x, G.b_y, G.b_li] = hl.mux(G.b_li,neighbors)

        cr = hl.RDom([(0, 8)])
        sum_neighbors = hl.Func("sum_neighbors")
        sum_neighbors[G.b_x, G.b_y] = hl.sum(hl.select(
            (cr==0) & ((block_ori_neighbors[G.b_x, G.b_y, cr]==7) | (block_ori_neighbors[G.b_x, G.b_y, cr]==105)),1,
            (cr==1) & ((block_ori_neighbors[G.b_x, G.b_y, cr]!=0) & ((block_ori_neighbors[G.b_x, G.b_y, cr]%5)==0)  & ((block_ori_neighbors[G.b_x, G.b_y, cr]%7)==0)),1,
            (cr==2) & ((block_ori_neighbors[G.b_x, G.b_y, cr]==5) | (block_ori_neighbors[G.b_x, G.b_y, cr]==70)),1,
            (cr==3) & ((block_ori_neighbors[G.b_x, G.b_y, cr]!=0) & ((block_ori_neighbors[G.b_x, G.b_y, cr]%3)==0)  & ((block_ori_neighbors[G.b_x, G.b_y, cr]%7)==0)),1,
            (cr==4) & ((block_ori_neighbors[G.b_x, G.b_y, cr]!=0) &((block_ori_neighbors[G.b_x, G.b_y, cr]%2)==0)  & ((block_ori_neighbors[G.b_x, G.b_y, cr]%5)==0)),1,
            (cr==5) & ((block_ori_neighbors[G.b_x, G.b_y, cr]==3) | (block_ori_neighbors[G.b_x, G.b_y, cr]==42)),1,
            (cr==6) & ((block_ori_neighbors[G.b_x, G.b_y, cr]!=0) & ((block_ori_neighbors[G.b_x, G.b_y, cr]%2)==0)  & ((block_ori_neighbors[G.b_x, G.b_y, cr]%3)==0)),1,
            (cr==7) & ((block_ori_neighbors[G.b_x, G.b_y, cr]==2) | (block_ori_neighbors[G.b_x, G.b_y, cr]==30)),1,
            (block_ori_neighbors[G.b_x, G.b_y, cr]==210), 1,
            0
        ))

        block_mask = hl.Func("block_mask")
        block_mask[G.b_x, G.b_y] = hl.select(
            (not_b_canny & (sum_neighbors[G.b_x, G.b_y]>0)), 1, 0
        )


        r_nd = hl.RDom([(0, 2),(0, 2)])
        fore_sum_new_depth = hl.Func("fore_sum_new_depth")
        fore_sum_new_depth[G.x, G.y] = hl.f32(hl.sum(
            hl.select(
                (block_n_bcanny_pixels[x_b_expr-r_nd.x, y_b_expr-r_nd.y]>0) & (first_labels_img[G.x,G.y]==2),
                1,
                0
            )
        ))


        initial_verts = hl.Func("initial_verts")
        x_coord = hl.f32(1.0) - (hl.f32(2.0*G.x)/(hl.f32(G.depth.width())))
        y_coord = hl.f32(1.0) - (hl.f32(2.0*G.y)/(hl.f32(G.depth.height())))
        z_coord = hl.f32(depth_clamped[G.x, G.y])/255.0
        initial_verts[G.x, G.y, G.c] = hl.select(
            G.c==0, x_coord,
            G.c==1, y_coord,
            z_coord
        )

        initial_uvs = hl.Func("initial_uvs")
        u_coord = (hl.f32(G.x)/(hl.f32(G.depth.width())))
        v_coord = hl.f32(1.0) - (hl.f32(G.y)/(hl.f32(G.depth.height())))
        initial_uvs[G.x, G.y, G.c] = hl.select(
            G.c==0, u_coord,v_coord
        )

        labels_img = hl.Func("labels_img")
        labels_img[G.x,G.y] = hl.select( x_block_check & y_block_check,
            hl.select(
                (hl.maximum(block_mask[x_b_expr-r_nd.x, y_b_expr-r_nd.y])>0) & (block_m_canny[x_b_expr, y_b_expr]==0) & (first_labels_img[G.x,G.y]==1),
                    3,
                first_labels_img[G.x,G.y]),
            0
        )


        X_expr = G.b_x*G.block_size
        Y_expr = G.b_y*G.block_size
        # block_depth = hl.Func("block_depth")
        # block_depth[G.b_x, G.b_y] = hl.f32(depth_clamped[X_expr, Y_expr])/255.0
        check_X = X_expr < G.canny.width()
        check_Y = Y_expr < G.canny.height()
        set_idx_X = hl.select(check_X, X_expr, G.canny.width()-1)
        set_idx_Y = hl.select(check_Y, Y_expr, G.canny.height()-1)
        G.BI_fore_Vcoords[G.b_x, G.b_y, G.c] = initial_verts[set_idx_X, set_idx_Y, G.c]
        G.BI_Vcoords[G.b_x, G.b_y, G.c] = initial_verts[set_idx_X, set_idx_Y, G.c]
        G.BI_Vcoords[G.b_x, G.b_y, 2] = hl.select(
            (labels_img[set_idx_X,set_idx_Y]>1),
                hl.minimum(
                    hl.f32(block_mean_depth[G.b_x-r_nd.x, G.b_y-r_nd.y])
                ),
            initial_verts[set_idx_X, set_idx_Y, 2]
        )

        G.BI_UVcoords[G.b_x, G.b_y, G.c] = initial_uvs[set_idx_X, set_idx_Y, G.c]


        # G.I_labels[G.x,G.y] = hl.u8(hl.select(labels_img[G.x,G.y]==2,
        #                                      2,
        #                                      labels_img[G.x,G.y]==1,
        #                                      1,
        #                                      0))
        G.I_labels[G.x,G.y] = hl.u8(labels_img[G.x, G.y])

        r_mask = hl.RDom([(-1, 3),(-1, 3)])
        r_mask.where((r_mask.x!=0) & (r_mask.y!=0))
        block_mask_bound = hl.BoundaryConditions.repeat_edge(block_mask, [(0,hl.i32(G.canny.width()/G.block_size)),(0,hl.i32(G.canny.height()/G.block_size))])
        G.BI_merge_mask[G.b_x, G.b_y] = hl.u8(
            hl.select(block_mask_bound[G.b_x, G.b_y]==1, 1,
                      hl.select(
                          (hl.maximum(block_mask_bound[G.b_x+r_mask.x, G.b_y+r_mask.y])==1) & (block_m_canny[G.b_x, G.b_y]==0),
                            2,
                            0
                          )
                      )
        )

        G.canny.set_estimates([(0,1280),(0,1280)])
        G.depth.set_estimates([(0,1280),(0,1280)])
        G.back_canny.set_estimates([(0,1280),(0,1280)])
        block_canny_labels.set_estimate(G.b_c, 0, 4)
        count_labels_img.set_estimate(G.l, 0, 2)
        initial_verts.set_estimate(G.c, 0, 2)
        G.I_labels.set_estimates([(0,1280),(0,1280)])
        G.BI_merge_mask.set_estimates([(0,41),(0,41)])
        G.BI_Vcoords.set_estimates([(0,41),(0,41),(0,3)])
        G.BI_fore_Vcoords.set_estimates([(0,41),(0,41),(0,3)])
        G.BI_UVcoords.set_estimates([(0,41),(0,41),(0,2)])
        G.block_stride_x.set_estimate(41)
        G.block_stride_y.set_estimate(41)
        if G.using_autoscheduler():
            # nothing
            pass
        # Schedule
        # v = G.natural_vector_size(hl.UInt(8))
        # G.output.vectorize(G.x, v)

if __name__ == "__main__":
    hl.main()