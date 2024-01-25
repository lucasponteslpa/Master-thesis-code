import halide as hl


@hl.alias(
    back_faces_Adams2019={"autoscheduler": "Adams2019"},
    back_faces_Mullapudi2016={"autoscheduler": "Mullapudi2016"},
    back_faces_Li2018={"autoscheduler": "Li2018"},
)

# Apply a mask value to a 2D image using a logical operator that is selected at compile-time.
@hl.generator(name = "back_faces")
class BackgroundFacesGenerator:
    block_size = hl.GeneratorParam(16)

    vert_img = hl.InputBuffer(hl.Float(32), 3)
    fore_vert_img = hl.InputBuffer(hl.Float(32), 3)
    uv_img = hl.InputBuffer(hl.Float(32), 3)
    label_img = hl.InputBuffer(hl.UInt(8), 2)

    out_verts = hl.OutputBuffer(hl.Float(32), 2)
    out_uvs = hl.OutputBuffer(hl.Float(32), 2)
    out_faces = hl.OutputBuffer(hl.Int(32),2)
    img_index = hl.OutputBuffer(hl.Int(32),2)

    i, i_f, c = hl.vars("i i_f c")
    x, y = hl.vars("x y")

    def generate(G):

        W = G.vert_img.width()
        H = G.vert_img.height()
        v0_x_idx = G.i%(W)
        v0_y_idx = hl.i32(hl.floor(hl.f32(G.i)/hl.f32(W)))
        # end_idx = (H*W-1)
        v1_x_idx = (G.i-(H*W))%(W)
        v1_y_idx = hl.i32(hl.floor(hl.f32(G.i-(H*W))/hl.f32(W)))
        vert_img_clamped = hl.BoundaryConditions.repeat_edge(G.vert_img)
        fore_vert_img_clamped = hl.BoundaryConditions.repeat_edge(G.fore_vert_img)
        G.out_verts[G.i, G.c] = hl.select(
            G.i<(W*H),
                vert_img_clamped[v0_x_idx, v0_y_idx, G.c],
            fore_vert_img_clamped[v1_x_idx, v1_y_idx, G.c]
        )

        uv_img_clamped = hl.BoundaryConditions.repeat_edge(G.uv_img)
        G.out_uvs[G.i, G.c] = hl.select(
            G.i<(W*H),
                uv_img_clamped[v0_x_idx, v0_y_idx, G.c],
            uv_img_clamped[v1_x_idx, v1_y_idx, G.c],
        )

        x_block_check = (G.x%G.block_size == 0)
        y_block_check = (G.y%G.block_size == 0)
        x_edge_check = (G.x==((G.block_size*(W-1))-1))
        y_edge_check = (G.y==((G.block_size*(H-1))-1))
        x_b_expr = hl.i32(hl.floor(G.x/G.block_size))
        x_e_expr = W-1
        y_b_expr = hl.i32(hl.floor(G.y/G.block_size))
        y_e_expr = H-1
        x_v_expr = hl.select(x_edge_check, x_e_expr, x_b_expr)
        y_v_expr = hl.select(y_edge_check, y_e_expr, y_b_expr)
        check_x_pixel = ((x_block_check | (y_block_check & x_edge_check)))
        check_y_pixel = ((y_block_check | (x_block_check & y_edge_check)))
        check_pixel = (check_x_pixel & check_y_pixel) | (x_edge_check & y_edge_check)
        G.img_index[G.x, G.y] = hl.select(
            check_pixel,
                hl.select(
                    G.label_img[G.x, G.y]<3,
                        x_v_expr + y_v_expr*W,
                    x_v_expr + y_v_expr*W + (W*H),
                    ),
            0
        )

        f_idx = hl.i32(hl.floor(hl.f32(G.i_f)/hl.f32(2.0)))
        v0_idx = f_idx
        v1_idx = f_idx + 1
        v2_idx = f_idx + W
        v3_idx = f_idx + W + 1
        check_y_pos = (hl.f32(v2_idx)/hl.f32(W)) < hl.f32(H)
        check_final = (((v1_idx%(W))!=0) | (f_idx==0)) & check_y_pos
        select_faces0 = hl.mux(G.c, [v0_idx, v1_idx, v2_idx])
        select_faces1 = hl.mux(G.c, [v1_idx, v2_idx, v3_idx])

        G.out_faces[G.i_f, G.c] = hl.select(
            check_final,
                hl.select(
                    (G.i_f%2)==0,
                        select_faces0,
                    select_faces1
                ),
            -1
        )

        G.vert_img.set_estimates([(0,41),(0,41),(0,3)])
        G.fore_vert_img.set_estimates([(0,41),(0,41),(0,3)])
        G.uv_img.set_estimates([(0,41),(0,41),(0,2)])
        G.label_img.set_estimates([(0,41*16),(0,41*16)])
        G.out_verts.set_estimates([(0,41*41),(0,3)])
        G.out_uvs.set_estimates([(0,41*41),(0,2)])
        G.out_faces.set_estimates([(0,2*40*40),(0,3)])
        G.img_index.set_estimates([(0,41*16),(0,41*16)])

        if G.using_autoscheduler():
            pass

if __name__ == "__main__":
    hl.main()