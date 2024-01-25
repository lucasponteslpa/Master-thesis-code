#include "Halide.h"
#include <stdio.h>
namespace {
using namespace Halide;
using namespace Halide::ConciseCasts;
using namespace Halide::Internal;

class ForegroundMeshVerts : public Halide::Generator<ForegroundMeshVerts> {
public:

    GeneratorParam<int> block_size{"block_size",16};

    Input<Buffer<uint8_t, 2>> I_canny{"I_canny"};
    Input<Buffer<uint8_t, 2>> I_back_canny{"I_back_canny"};
    Input<Buffer<uint8_t, 2>> I_depth{"I_depth"};

    Output<Buffer<int32_t, 3>> P_fore_faces{"P_fore_faces"};
    Output<Buffer<int32_t, 1>> P_Nfaces{"P_Nfaces"};
    Output<Buffer<int32_t, 2>> I_Vlabels{"I_Vlabels"};
    Output<Buffer<int32_t, 2>> I_ForeMask{"I_ForeMask"};

    void generate() {
        Expr w = I_canny.dim(0).extent();
        Expr h = I_canny.dim(1).extent();

        Expr w_b = i32(floor(f32(w)/block_size));
        Expr h_b = i32(floor(f32(h)/block_size));

        canny_bound = BoundaryConditions::repeat_edge(I_canny, {{0, w}, {0, h}});
        b_canny_bound = BoundaryConditions::repeat_edge(I_back_canny, {{0, w}, {0, h}});
        depth_bound = BoundaryConditions::repeat_edge(I_depth, {{0, w}, {0, h}});

        Expr pX_img =  clamp(((p*block_size) % (w)) + x_p, 0, w-1);
        Expr pY_img =  clamp(i32(floor(f32(p) / f32(w_b))*block_size) + y_p, 0, h-1);

        patched_canny( x_p, y_p, p) = canny_bound(pX_img, pY_img);
        patched_b_canny( x_p, y_p, p) = b_canny_bound(pX_img, pY_img);
        patched_depth( x_p, y_p, p) = depth_bound(pX_img, pY_img);
        patch_coords( x_p, y_p, p) = pX_img + pY_img*w;

        r_p = RDom( 0, block_size+1, 0, block_size+1);

        Expr check_canny = patched_canny(r_p.x, r_p.y, p)>0;
        Expr check_b_canny = patched_b_canny(r_p.x, r_p.y, p)>0;

        sum_p_canny(p) = sum(select(check_canny, 1.0f, 0.0f));
        sum_p_b_canny(p) = sum(select(check_b_canny, 1.0f, 0.0f));

        Expr select_div = select(sum_p_canny(p)>0.0f, sum_p_canny(p), 1.0f);
        Expr select_b_div = select(sum_p_b_canny(p)>0.0f, sum_p_b_canny(p), 1.0f);

        mean_depth(p) = f32(sum(select(check_canny && sum_p_canny(p)>0.0f, f32(patched_depth(r_p.x, r_p.y, p)), 0)))/select_div;
        mean_b_depth(p) = f32(sum(select(check_b_canny && sum_p_b_canny(p)>0.0f, f32(patched_depth(r_p.x, r_p.y, p)), 0)))/select_b_div;

        labeled.define_extern(
            "search_quadtree",
            {
                patched_canny,
                mean_depth,
                mean_b_depth,
                patched_depth,
                patch_coords,
                i32(block_size)
            },
            {Int(32), Int(32)},
            {x_p, y_p, p},
            NameMangling::C
        );
        Expr proxy_expr = patched_canny(0, 0, p) + patched_canny(block_size, block_size, p)
                          + patched_depth(0, 0, p) + patched_depth(block_size, block_size, p)
                          + patch_coords(0, 0, p) + patch_coords(block_size, block_size, p)
                          + mean_depth(p) + mean_b_depth(p);
        labeled.function().extern_definition_proxy_expr() = (proxy_expr,proxy_expr);

        Expr img_xp = x%(block_size);
        Expr img_yp = y%(block_size);
        Expr imgX_p  = i32(floor(f32(x)/(block_size)));
        Expr imgY_p  = i32(floor(f32(y)/(block_size)));
        Expr img_p = imgY_p*w_b + imgX_p;

        Expr check_xp = (img_xp == 0) && ((x)>0 && x<(w-1));
        Expr check_yp = (img_yp == 0) && ((y)>0 && y<(h-1));
        Expr check_p = ((img_p>0) && (img_p<((w_b*h_b)-1)));
        Expr up_idx = max(0,img_p-w_b);
        Expr back_idx = max(0,img_p-1);
        Expr back_up_idx = max(0, img_p-w_b-1);
        Expr prev_l = select(
            check_xp && !check_yp && check_p,
                select(
                    (labeled( 0, img_yp, img_p)[0]>0) || (labeled( 16, img_yp, back_idx)[0]>0),1,0
                ),
            !check_xp && check_yp && check_p,
                select(
                    (labeled( img_xp, 0, img_p)[0]>0) || (labeled( img_xp, 16, up_idx)[0]>0),1,0
                ),
            check_xp && check_yp && check_p,
                select(
                    (labeled( 0, 0, img_p)[0]>0) || (labeled( 0, 16, up_idx)[0]>0) || (labeled( 16, 16, back_up_idx)[0]>0) || (labeled( 16, 0, back_idx)[0]>0),
                    1,
                    0
                ),
            labeled( img_xp, img_yp, img_p)[0]

        );

        Expr prev_m = select(
            check_xp && !check_yp && check_p,
                select(
                    (labeled( 0, img_yp, img_p)[1]>0) || (labeled( 16, img_yp, back_idx)[1]>0),1,0
                ),
            !check_xp && check_yp && check_p,
                select(
                    (labeled( img_xp, 0, img_p)[1]>0) || (labeled( img_xp, 16, up_idx)[1]>0),1,0
                ),
            check_xp && check_yp && check_p,
                select(
                    (labeled( 0, 0, img_p)[1]>0) || (labeled( 0, 16, up_idx)[1]>0) || (labeled( 16, 16, back_up_idx)[1]>0) || (labeled( 16, 0, back_idx)[1]>0),
                    1,
                    0
                ),
            labeled( img_xp, img_yp, img_p)[1]

        );
        I_Vlabels(x, y) = prev_l;
        I_ForeMask(x, y) = prev_m;
        out_labeled(x_p, y_p, p) = I_Vlabels(pX_img, pY_img);


        patch_faces.define_extern(
            "search_quadtree_faces",
            {
                patched_canny,
                patch_coords,
                out_labeled,
                i32(block_size)
            },
            {Int(32)},
            {c, f, p},
            NameMangling::C
        );
        Expr f_proxy_expr = patched_canny(0, 0, p) + patched_canny(block_size, block_size, p)
                          + patch_coords(0, 0, p) + patch_coords(block_size, block_size, p)
                          + out_labeled(0,0,p) + out_labeled(block_size,block_size,p);
        patch_faces.function().extern_definition_proxy_expr() = f_proxy_expr;

        P_fore_faces(c, f, p) = patch_faces(c, f, p);

        r_f = RDom( 0, 2*(block_size+1)*(block_size+1));
        P_Nfaces(p) = sum(select((patch_faces(0, r_f, p)>=0) && (patch_faces(1, r_f, p)>=0) && (patch_faces(2, r_f, p)>=0),1,0));

    }

    void schedule() {
        int vector_size = natural_vector_size(UInt(8));

        if(using_autoscheduler()) {
            I_canny.set_estimates({{0, 656},{0, 656}});
        } else {
            Var po,pi,xo,xi,yo,yi,xy;
            int split_size = 32;
            I_Vlabels.compute_root()
                     .split(x,xo,xi,split_size)
                     .split(y,yo,yi,split_size)
                     .reorder(xi,yi,xo,yo)
                     .fuse(xo,yo,xy)
                     .parallel(xy);
            I_ForeMask.compute_root()
                     .split(x,xo,xi,split_size)
                     .split(y,yo,yi,split_size)
                     .reorder(xi,yi,xo,yo)
                     .fuse(xo,yo,xy)
                     .parallel(xy)
                     .compute_with(I_Vlabels, xy);
            P_Nfaces.compute_root().parallel(p);
            out_labeled.compute_root();
            P_fore_faces.compute_root()
                    .split(p, po, pi, split_size)
                    .parallel(po);
            labeled.compute_root()
                .split(p, po, pi, split_size)
                .parallel(po)
                ;
            patch_faces.compute_root()
                .split(p, po, pi, split_size)
                .parallel(po)
                ;
            patched_canny.compute_root()
                    .split(p, po, pi, split_size)
                    .parallel(po)
                    ;
            patched_b_canny.compute_root()
                    .split(p, po, pi, split_size)
                    .parallel(po)
                    .compute_with(patched_canny, po)
                    ;
            patched_depth.compute_root()
                    .split(p, po, pi, split_size)
                    .parallel(po)
                    .compute_with(patched_canny, po)
                    ;
            patch_coords.compute_root()
                    .split(p, po, pi, split_size)
                    .parallel(po)
                    .compute_with(patched_canny, po)
                    ;
            sum_p_canny.compute_root()
                    .split(p, po, pi, split_size)
                    .parallel(po)
                    ;
            sum_p_b_canny.compute_root()
                    .split(p, po, pi, split_size)
                    .parallel(po)
                    .compute_with(sum_p_canny, po)
                    ;
            mean_depth.compute_root()
                    .split(p, po, pi, split_size)
                    .parallel(po)
                    ;
            mean_b_depth.compute_root()
                    .split(p, po, pi, split_size)
                    .parallel(po)
                    .compute_with(mean_depth, po)
                    ;
        }

    }

private:
    Var x{"x"}, y{"y"}, x_p{"x_p"}, y_p{"y_p"}, p{"p"}, a{"a"}, c{"c"},f{"f"};
    Func  canny_bound{"canny_bound"}, b_canny_bound{"b_canny_bound"}, depth_bound{"depth_bound"},
    patched_b_canny{"patched_b_canny"}, patched_depth{"patched_depth"}, patched_canny{"patched_canny"},
    labeled{"labeled"}, out_labeled{"out_labeled"},sum_p_canny{"sum_p_canny"}, sum_p_b_canny{"sum_p_b_canny"}, mean_depth{"mean_depth"},
    mean_b_depth{"mean_b_depth"}, patch_coords{"patch_coords"}, patch_init_faces{"patch_init_faces"},
    patch_faces{"patch_faces"};
    RDom r_p, rm_p, r_f;
};

}  // namespace

HALIDE_REGISTER_GENERATOR(ForegroundMeshVerts, foreground_mesh_verts)
