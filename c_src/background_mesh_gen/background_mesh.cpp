#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <math.h>
#include <stdio.h>

#include "HalideBuffer.h"
#include "halide_image_io.h"
#include "halide_benchmark.h"
#include "halide_malloc_trace.h"

#include "background_mesh_verts.h"
#include "background_mesh_faces.h"

using namespace Halide::Runtime;
using namespace Halide::Tools;

int main(int argc, char **argv) {
    if (argc < 4) {
        printf("Usage: ./process canny.png back_canny.png depth.png\n");
        return 0;
    }

    const char * canny_filename = argv[1];
    const char * back_canny_filename = argv[2];
    const char * depth_filename = argv[3];


    Buffer<uint8_t> in_canny = load_and_convert_image(canny_filename);
    Buffer<uint8_t> in_back_canny = load_and_convert_image(back_canny_filename);
    Buffer<uint8_t> in_depth = load_and_convert_image(depth_filename);

    int W = in_canny.width(), H = in_canny.height();
    int block_size = 16;
    int w_p = std::floor(((float)W)/block_size);
    int h_p = std::floor(((float)H)/block_size);
    printf("%d %d\n", W, H);

    Buffer<uint8_t> canny(W, H);
    Buffer<uint8_t> back_canny(W, H);
    Buffer<uint8_t> depth(W, H);

    int n_patches = ((w_p)*(h_p));
    int total_n_faces = 2*block_size*block_size;
    Buffer<uint8_t> out_mask_bimg(w_p+1, h_p+1);
    Buffer<float> out_verts_bimg(w_p+1, h_p+1, 3);
    Buffer<float> out_fore_verts_bimg(w_p+1, h_p+1, 3);
    Buffer<float> out_uvs_bimg(w_p+1, h_p+1, 3);
    Buffer<uint8_t> out_label_img(W, H);

    canny.for_each_element([&](int x, int y) {
        canny(x, y) = (in_canny(x,y,0));
    });

    back_canny.for_each_element([&](int x, int y) {
        back_canny(x, y) = (in_back_canny(x,y,0));
    });

    depth.for_each_element([&](int x, int y) {
        depth(x, y) = (in_depth(x,y,0));
    });



    BenchmarkResult time = benchmark([&]() {
        background_mesh_verts(canny, back_canny, depth, W, H, out_mask_bimg, out_verts_bimg, out_fore_verts_bimg, out_uvs_bimg, out_label_img);
    });
    printf("vertices execution time: %lf ms\n", time * 1e3);
    printf("number of patched: %d\n", (w_p)*(h_p));

    Buffer<float> out_verts(2*(w_p + 1)*(h_p + 1),3);
    Buffer<float> out_uvs(2*(w_p + 1)*(h_p + 1),3);
    Buffer<int32_t> out_faces(2*(w_p )*(h_p),3);
    Buffer<int32_t> out_idx_img(W, H);

    time = benchmark([&]() {
        background_mesh_faces(out_verts_bimg, out_fore_verts_bimg, out_uvs_bimg, out_label_img, out_verts, out_uvs, out_faces, out_idx_img);
    });

    printf("faces execution time: %lf ms\n", time * 1e3);

    return 0;
}
