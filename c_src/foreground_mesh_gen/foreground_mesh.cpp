#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <math.h>
#include <stdio.h>

#include "HalideBuffer.h"
#include "halide_image_io.h"
#include "halide_benchmark.h"
#include "halide_malloc_trace.h"

#include "foreground_mesh_verts.h"
#include "foreground_mesh_faces.h"

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
    Buffer<int32_t> out_patch_faces(3, total_n_faces, n_patches);
    Buffer<int32_t> out_N_faces(n_patches);
    Buffer<int32_t> out_label_img(W, H);
    Buffer<int32_t> out_mask_img(W, H);

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
        foreground_mesh_verts(canny, back_canny, depth, out_patch_faces, out_N_faces, out_label_img, out_mask_img);
    });
    printf("vertices execution time: %lf ms\n", time * 1e3);
    printf("number of patched: %d\n", (w_p)*(h_p));

    int NV = 0;
    out_label_img.for_each_element([&](int x, int y) {
        if(out_label_img(x,y)>0) NV++;
    });

    int NF = 0;
    out_N_faces.for_each_element([&](int f) {
        NF += out_N_faces(f);
    });

    Buffer<float> out_verts(3,NV);
    Buffer<float> out_uv(3, NV);
    Buffer<int32_t> out_faces(3, NF);
    Buffer<int32_t> out_idx_img(W, H);
    out_idx_img.for_each_element([&](int x, int y) {
        out_idx_img(x,y) = -1;
    });
    printf("NF %d\n", NF);
    printf("NV %d\n", NV);
    time = benchmark([&]() {
        foreground_mesh_faces(depth, out_patch_faces, out_label_img, 0, out_idx_img, out_verts, out_uv, out_faces);
    });

    printf("faces execution time: %lf ms\n", time * 1e3);

    return 0;
}
