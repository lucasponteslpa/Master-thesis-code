#include "Halide.h"
#include <stdio.h>
namespace {
using namespace Halide;
using namespace Halide::ConciseCasts;
using namespace Halide::Internal;

class DepthEnhance : public Halide::Generator<DepthEnhance> {
public:

    GeneratorParam<int> block_size{"block_size",32};
    GeneratorParam<int> stride{"stride",32};


    Input<Buffer<uint8_t, 2>> I_depth{"I_depth"};
    Input<Buffer<uint8_t, 3>> I_masks{"I_masks"};

    Output<Buffer<float, 2>> out_depth{"out_depth"};
    Output<Buffer<float, 3>> depth_enh{"depth_enh"};
    Output<Buffer<float, 3>> depth_blocks{"depth_blocks"};

    Expr gaussian(Expr i, Expr sigma) {
            return exp(-(i*i)/(2.f*sigma));
    }

    Expr bilinear(
        Expr p_x,
        Expr p_y,
        Expr x1,
        Expr y1,
        Expr x2,
        Expr y2,
        Func Q
    ){

        Expr Q11 = f32(Q(i32(x1),i32(y1),c));
        Expr Q21 = f32(Q(i32(x2),i32(y1),c));
        Expr Q12 = f32(Q(i32(x1),i32(y2),c));
        Expr Q22 = f32(Q(i32(x2),i32(y2),c));
        Expr f_1 = ((x2-p_x)/(x2-x1))*Q11 + ((p_x - x1)/(x2-x1))*Q21;
        Expr f_2 = ((x2-p_x)/(x2-x1))*Q12 + ((p_x - x1)/(x2-x1))*Q22;
        Expr f = ((y2-p_y)/(y2-y1))*f_1 + ((p_y - y1)/(y2-y1))*f_2;
        return f;
    }

    void generate() {
        Expr w = I_depth.dim(0).extent();
        Expr w_b = I_depth.dim(0).extent()/(stride);
        Expr h = I_depth.dim(1).extent();
        Expr h_b = I_depth.dim(1).extent()/(stride);
        Expr n = I_masks.dim(2).extent();
        depth_bound = BoundaryConditions::repeat_edge(I_depth, {{0, w}, {0, h}});
        masks_bound = BoundaryConditions::repeat_edge(I_masks, {{0, w}, {0, h}, {0,n}});


        r_p = RDom(0,block_size,0,block_size);
        Expr pX_r =  clamp(((p*(stride)) % (w)) + r_p.x, 0, w-1);
        Expr pY_r =  clamp(i32(floor(f32(p) / f32(w_b))*(stride)) + r_p.y, 0, h-1);

        Expr pX_img =  clamp(((p*(stride)) % (w)) + x_p, 0, w-1);
        Expr pY_img =  clamp(i32(floor(f32(p) / f32(w_b))*(stride)) + y_p, 0, h-1);

        Expr pX =  clamp(((p*(stride)) % (w)), 0, w-1);
        Expr pY =  clamp(i32(floor(f32(p) / f32(w_b))*(stride)), 0, h-1);


        masks_block_sum(p, c) = sum(i32(masks_bound(pX_r, pY_r, c)));

        r_img = RDom(0,w,0,h);
        masks_sum(c) = sum(i32(masks_bound(r_img.x, r_img.y, c)));
        area_masks(x,y,c) = i32(masks_bound(x,y,c))*masks_sum(c);

        masks_depth_mean(p, c) = select(
            masks_block_sum(p, c)>0 ,
                sum(i32(masks_bound(pX_r, pY_r, c)*depth_bound(pX_r, pY_r)))/masks_block_sum(p, c),
                0
        );
        r_c = RDom(0,n);
        masks_sum_c(x,y) = sum(masks_bound(x,y,r_c));


        patch_coords_x( x_p, y_p, p) = pX_img;
        patch_coords_y( x_p, y_p, p) = pY_img;

        depth_samples(p, q) = select(
            q==0,
                depth_bound(pX,pY),
            q==1,
                depth_bound(pX+block_size,pY),
            q==2,
                depth_bound(pX,pY+block_size),
            depth_bound(pX+block_size,pY+block_size)
        );

        masks_samples(p, c, q) = select(
            q==0,
                masks_bound(pX,pY,c),
            q==1,
                masks_bound(pX+block_size,pY,c),
            q==2,
                masks_bound(pX,pY+block_size,c),
            masks_bound(pX+block_size,pY+block_size,c)
        );

        r_s = RDom(0,4);
        masks_samples_sum(p, c) = sum(i32(masks_samples(p,c,r_s)));
        mean_depth_samples(p, c) = maximum(
            select(
                masks_samples_sum(p,c) > 0,
                    i32(depth_samples(p, r_s)*masks_samples(p,c,r_s)),
                0
            )
        );

        masks_depth_corners(p, c, q) = mean_depth_samples(p,c);

        r_b = RDom( 0, block_size+1, 0, block_size+1, 0, w_b*h_b);
        Expr p_x = patch_coords_x(r_b.x,r_b.y,r_b.z);
        Expr p_y = patch_coords_y(r_b.x,r_b.y,r_b.z);
        Expr x1 = patch_coords_x(0,0,r_b.z);
        Expr y1 = patch_coords_y(0,0,r_b.z);
        Expr x2 = patch_coords_x(block_size,block_size,r_b.z);
        Expr y2 = patch_coords_y(block_size,block_size,r_b.z);

        Expr comp_r0 = (r_b.x==0) && (r_b.y==0);
        Expr comp_r1 = (r_b.x==block_size) && (r_b.y==0);
        Expr comp_r2 = (r_b.x==0) && (r_b.y==block_size);
        Expr comp_r3 = (r_b.x==block_size) && (r_b.y==block_size);
        n_corners_img(x, y, c) = 0.0f;
        n_corners_img(p_x, p_y, c) += f32(select(
            comp_r0,
                // masks_samples(r_b.z, c, 0),
                select(masks_samples_sum(r_b.z, c)>0,1,0),
            comp_r1,
                // masks_samples(r_b.z, c, 1),
                select(masks_samples_sum(r_b.z, c)>0,1,0),
            comp_r2,
                // masks_samples(r_b.z, c, 2),
                select(masks_samples_sum(r_b.z, c)>0,1,0),
            comp_r3,
                // masks_samples(r_b.z, c, 3),
                select(masks_samples_sum(r_b.z, c)>0,1,0),
            0
        ));

        depth_corners_img(x, y, c) = 0.0f;
        depth_corners_img(p_x, p_y, c) += select(
            comp_r0,
                f32(masks_depth_corners(r_b.z, c, 0))/f32(clamp(n_corners_img(p_x, p_y, c),1.0f,f32(n))),
            comp_r1,
                f32(masks_depth_corners(r_b.z, c, 1))/f32(clamp(n_corners_img(p_x, p_y, c),1.0f,f32(n))),
            comp_r2,
                f32(masks_depth_corners(r_b.z, c, 2))/f32(clamp(n_corners_img(p_x, p_y, c),1.0f,f32(n))),
            comp_r3,
                f32(masks_depth_corners(r_b.z, c, 3))/f32(clamp(n_corners_img(p_x, p_y, c),1.0f,f32(n))),
            0.0f
        );

        depth_enh(x,y,c) = 0.0f;
        depth_enh(p_x,p_y,c) = masks_bound(p_x,p_y,c)*bilinear(f32(p_x), f32(p_y), f32(x1), f32(y1), f32(x2), f32(y2), depth_corners_img);
        max_area_masks(x,y) = maximum(area_masks(x,y,r_c));
        out_depth(x,y) = select(
            masks_sum_c(x,y)>0,
                maximum(
                    select(
                        max_area_masks(x,y)==area_masks(x,y,r_c),
                            depth_enh(x,y,r_c),
                        0.0f
                    )
                ),
            f32(depth_bound(x,y))
        );

        depth_blocks(x_p,y_p,p) = f32(patch_coords_x( x_p, y_p, p));

    }

    void schedule() {
        int vector_size = natural_vector_size(UInt(8));

        if(using_autoscheduler()) {
            I_depth.set_estimates({{0, 1280},{0, 1280}});
            I_masks.set_estimates({{0,60},{0, 1280},{0, 1280}});
            out_depth.set_estimates({{0, 1280},{0, 1280}});
            depth_enh.set_estimates({{0,60}, {0, 1280},{0, 1280}});
            depth_blocks.set_estimates({{0, 16},{0, 16}, {0,41*41}});

        } else {
            masks_block_sum.compute_root();
            masks_depth_mean.compute_root();
            masks_sum_c.compute_root();
            masks_sum.compute_root();
            area_masks.compute_root();
            patch_coords_x.compute_root();
            patch_coords_y.compute_root();
            depth_samples.compute_root();
            masks_samples.compute_root();
            masks_samples_sum.compute_root();
            mean_depth_samples.compute_root();
            masks_depth_corners.compute_root();
            n_corners_img.compute_root();
            depth_corners_img.compute_root();
            depth_enh.compute_root();
            depth_blocks.compute_root();
            max_area_masks.compute_root();
        }

    }

private:
    Var x{"x"}, y{"y"}, x_p{"x_p"}, y_p{"y_p"}, p{"p"}, c{"c"},f{"f"}, i{"i"}, j{"j"}, q{"q"};
    Func  depth_bound{"depth_bound"},masks_bound{"masks_bound"},I_vert_idx{"I_vert_idx"},
    vert_arr{"vert_arr"}, face_arr{"face_arr"}, masks_depth_mean{"masks_depth_mean"}, masks_block_sum{"masks_block_sum"},
    interp_kernel{"interp_kernel"}, kernels{"kernels"}, blocks_coords{"blocks_coords"},
    patch_coords_x{"pathc_coords_x"}, patch_coords_y{"pathc_coords_y"}, sum_kernel{"sum_kernel"},
    masks_sum_c{"masks_sum_c"}, depth_samples{"depth_samples"}, masks_samples{"masks_samples"},
    masks_samples_sum{"masks_samples_sum"}, mean_depth_samples{"mean_depth_samples"},
    masks_depth_corners{"masks_depth_corners"}, depth_corners_img{"depth_corners_img"},
    n_corners_img{"n_corners_img"}, area_masks{"area_masks"}, pixel_masks{"pixel_masks"},
    masks_sum{"masks_sum"}, max_area_masks{"max_area_masks"}, reference_mask{"reference_mask"};
    RDom r_p, rm_p, r_c, r_b, r_s, r_img;
};

}  // namespace

HALIDE_REGISTER_GENERATOR(DepthEnhance, depth_enhance)
