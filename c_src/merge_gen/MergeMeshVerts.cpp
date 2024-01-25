#include "Halide.h"
#include <stdio.h>
namespace {
using namespace Halide;
using namespace Halide::ConciseCasts;
using namespace Halide::Internal;

class MergeMeshVerts : public Halide::Generator<MergeMeshVerts> {
public:

    GeneratorParam<int> block_size{"block_size",16};

    Input<Buffer<int32_t, 2>> I_fore_idx{"I_fore_idx"};
    Input<Buffer<int32_t, 2>> I_back_idx{"I_back_idx"};
    Input<Buffer<uint8_t, 2>> I_back_labels{"I_back_labels"};
    // Input<Buffer<uint8_t, 2>> I_canny_mask{"I_canny_mask"};
    Input<Buffer<uint8_t, 2>> BI_fore_mask{"BI_fore_mask"};

    Input<int32_t> N0_verts{"N0_verts"};

    Output<Buffer<int32_t, 3>> P_out_faces{"P_out_faces"};
    Output<Buffer<int32_t, 1>> P_N_faces{"P_N_faces"};
    Output<Buffer<int32_t, 2>> I_out_vidx{"I_out_vidx"};


    void generate() {
        Expr w = I_fore_idx.dim(0).extent();
        Expr h = I_fore_idx.dim(1).extent();

        Expr w_b = BI_fore_mask.dim(0).extent()-1;
        Expr h_b = BI_fore_mask.dim(1).extent()-1;

        Expr check_x = ((x-(block_size/2))%block_size)==0;
        Expr check_y = ((y-(block_size/2))%block_size)==0;
        Expr X_b = i32(floor(f32(x)/f32(block_size)));
        Expr Y_b = i32(floor(f32(y)/f32(block_size)));
        Expr check_mask = BI_fore_mask(X_b,Y_b)==1;
        // I_vidxs(x,y) = select(
        //     check_mask & check_x & check_y,
        //         1,
        //     0
        // );

        Expr pX_img =  clamp(((p*block_size) % (w)) + x_p, 0, w-1);
        Expr pY_img =  clamp(i32(floor(f32(p) / f32(w_b))*block_size) + y_p, 0, h-1);

        patch_I_f_idx(x_p, y_p, p) = I_fore_idx(pX_img, pY_img);
        patch_I_b_idx(x_p, y_p, p) = I_back_idx(pX_img, pY_img);
        patch_I_b_l(x_p, y_p, p) = I_back_labels(pX_img, pY_img);
        patch_coords( x_p, y_p, p) = pX_img + pY_img*w;

        Expr pX_b = p%w_b;
        Expr pY_b = p/w_b;
        patch_f_mask(p) = BI_fore_mask(pX_b, pY_b);
        patch_vidx.define_extern(
            "load_vert_idx",
            {
                patch_f_mask,
                i32(N0_verts),
            },
            {Int(32)},
            {p},
            NameMangling::C
        );
        Expr vidx_proxy = patch_f_mask(0) + patch_f_mask(w_b*h_b-1);
        patch_vidx.function().extern_definition_proxy_expr() = vidx_proxy;

        Expr img_xp = x%(block_size);
        Expr img_yp = y%(block_size);
        Expr imgX_p  = i32(floor(f32(x)/(block_size)));
        Expr imgY_p  = i32(floor(f32(y)/(block_size)));
        Expr img_p = imgY_p*w_b + imgX_p;
        // Expr check_mask = BI_fore_mask(X_b,Y_b)==1;
        I_out_vidx(x,y) = select(
            check_mask & check_x & check_y,
                patch_vidx(img_p),
            -1
        );

        get_faces.define_extern(
            "search_mid_faces",
            {
                patch_I_f_idx,
                patch_I_b_idx,
                patch_I_b_l,
                patch_f_mask,
                patch_vidx,
                i32(block_size)
            },
            {Int(32)},
            {c, f, p},
            NameMangling::C
        );
        Expr proxy_expr = patch_I_f_idx(0, 0, p) + patch_I_f_idx(block_size, block_size, p)
                          + patch_I_b_idx(0, 0, p) + patch_I_b_idx(block_size, block_size, p)
                          + patch_I_b_l(0, 0, p) + patch_I_b_l(block_size, block_size, p)
                          + patch_vidx(p) + patch_f_mask(p);
        get_faces.function().extern_definition_proxy_expr() = proxy_expr;
        P_out_faces(c, f, p) = get_faces(c, f, p);

        r_f = RDom( 0, (block_size+1)*(block_size+1));
        P_N_faces(p) = sum(select((P_out_faces(0, r_f, p)>=0) && (P_out_faces(1, r_f, p)>=0) && (P_out_faces(2, r_f, p)>=0),1,0));
        // P_N_faces(p) = sum(select((patch_f_mask(p)==2) && (P_out_faces(0, r_f, p)>=0) && (P_out_faces(1, r_f, p)>=0) && (P_out_faces(2, r_f, p)>=0),1,0));
        // P_N_faces(p) = patch_vidx(p);
        // P_N_faces(p) = i32(patch_f_mask(p));

    }

    void schedule() {
        int vector_size = natural_vector_size(UInt(8));

        if(using_autoscheduler()) {
            I_fore_idx.set_estimates({{0, 656},{0, 656}});
        } else {
            Var po,pi;
            int split_size = 32;
            I_out_vidx.compute_root().parallel(y);
            patch_I_f_idx.compute_root()
                .split(p, po, pi, split_size)
                .parallel(po)
                ;
            patch_I_b_idx.compute_root()
                .split(p, po, pi, split_size)
                .parallel(po)
                ;
            patch_vidx.compute_root()
                ;
            patch_I_b_l.compute_root()
                .split(p, po, pi, split_size)
                .parallel(po)
                ;
            patch_f_mask.compute_root()
                .split(p, po, pi, split_size)
                .parallel(po)
                ;
            patch_coords.compute_root()
                .split(p, po, pi, split_size)
                .parallel(po)
                ;
            get_faces.compute_root()
                .split(p, po, pi, split_size)
                .parallel(po)
                ;
            P_out_faces.compute_root()
                .split(p, po, pi, split_size)
                .parallel(po)
                ;
            P_N_faces.compute_root()
                .split(p, po, pi, split_size)
                .parallel(po)
                ;
        }

    }

private:
    Var x{"x"}, y{"y"}, x_p{"x_p"}, y_p{"y_p"}, p{"p"}, a{"a"}, c{"c"},f{"f"};
    Func  I_vidxs{"I_vidxs"},patch_I_f_idx{"patch_I_f_idx"}, patch_I_b_idx{"patch_I_b_idx"}, patch_I_b_l{"patch_I_b_l"},
    patch_vidx{"patch_vidx"}, patch_f_mask{"patch_f_mask"}, patch_coords{"patch_coords"}, get_faces{"get_faces"};
    RDom r_p, rm_p, r_f;
};

}  // namespace

HALIDE_REGISTER_GENERATOR(MergeMeshVerts, merge_mesh_verts)
