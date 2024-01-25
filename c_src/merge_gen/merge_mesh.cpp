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

#include "background_mesh_verts.h"
#include "background_mesh_faces.h"

#include "merge_mesh_verts.h"
#include "merge_mesh_faces.h"

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
    int n_patches = ((w_p)*(h_p));
    int total_n_faces = 2*block_size*block_size;
    printf("%d %d\n", W, H);

    Buffer<uint8_t> canny(W, H);
    Buffer<uint8_t> back_canny(W, H);
    Buffer<uint8_t> depth(W, H);

    canny.for_each_element([&](int x, int y) {
        canny(x, y) = (in_canny(x,y,0));
    });

    back_canny.for_each_element([&](int x, int y) {
        back_canny(x, y) = (in_back_canny(x,y,0));
    });

    depth.for_each_element([&](int x, int y) {
        depth(x, y) = (in_depth(x,y,0));
    });

    /*///////////////////////////////////////////////
    /////////////// BACKGROUND MESH /////////////////
    ///////////////////////////////////////////////*/

    Buffer<uint8_t> back_mask_bimg(w_p+1, h_p+1);
    Buffer<float> back_verts_bimg(w_p+1, h_p+1, 3);
    Buffer<float> back_fore_verts_bimg(w_p+1, h_p+1, 3);
    Buffer<float> back_uvs_bimg(w_p+1, h_p+1, 3);
    Buffer<uint8_t> back_label_img(W, H);

    background_mesh_verts(canny, back_canny, depth, W, H, back_mask_bimg, back_verts_bimg, back_fore_verts_bimg, back_uvs_bimg, back_label_img);

    int n_back_verts = 2*(w_p + 1)*(h_p + 1);
    Buffer<float> back_verts(n_back_verts,3);
    Buffer<float> back_uvs(n_back_verts,3);
    Buffer<int32_t> back_faces(2*(w_p )*(h_p),3);
    Buffer<int32_t> back_idx_img(W, H);

    background_mesh_faces(back_verts_bimg, back_fore_verts_bimg, back_uvs_bimg, back_label_img, back_verts, back_uvs, back_faces, back_idx_img);


    /*///////////////////////////////////////////////
    /////////////// FOREGROUND MESH /////////////////
    ///////////////////////////////////////////////*/

    Buffer<int32_t> fore_patch_faces(3, total_n_faces, n_patches);
    Buffer<int32_t> fore_N_faces(n_patches);
    Buffer<int32_t> fore_label_img(W, H);
    Buffer<int32_t> fore_mask_img(W, H);


    foreground_mesh_verts(canny, back_canny, depth, fore_patch_faces, fore_N_faces, fore_label_img, fore_mask_img);

    int NV = 0;
    fore_label_img.for_each_element([&](int x, int y) {
        if(fore_label_img(x,y)>0) NV++;
    });

    int NF = 0;
    fore_N_faces.for_each_element([&](int f) {
        NF += fore_N_faces(f);
    });

    Buffer<float> fore_verts(3,NV);
    Buffer<float> fore_uvs(3, NV);
    Buffer<int32_t> fore_faces(3, NF);
    Buffer<int32_t> fore_idx_img(W, H);
    fore_idx_img.for_each_element([&](int x, int y) {
        fore_idx_img(x,y) = -1;
    });

    foreground_mesh_faces(depth, fore_patch_faces, fore_label_img, n_back_verts, fore_idx_img, fore_verts, fore_uvs, fore_faces);

    /*///////////////////////////////////////////////
    ////////////////// MERGE MESH ///////////////////
    ///////////////////////////////////////////////*/

    Buffer<int32_t> merge_patch_faces(3, total_n_faces, n_patches);
    Buffer<int32_t> merge_N_faces(n_patches);
    Buffer<int32_t> merge_label_img(W, H);

    BenchmarkResult time = benchmark([&]() {
        merge_mesh_verts(
            fore_idx_img,
            back_idx_img,
            back_label_img,
            back_mask_bimg,
            NV+n_back_verts,
            merge_patch_faces,
            merge_N_faces,
            merge_label_img
        );
    });
    printf("merge vertices execution time: %lf ms\n", time * 1e3);

    NV = 0;
    merge_label_img.for_each_element([&](int x, int y) {
        if(merge_label_img(x,y)>0) NV++;
    });

    NF = 0;
    merge_N_faces.for_each_element([&](int f) {
        NF += merge_N_faces(f);
    });

    Buffer<float> merge_verts(3,NV);
    Buffer<float> merge_uvs(3, NV);
    Buffer<int32_t> merge_faces(3, NF);

    time = benchmark([&]() {
        merge_mesh_faces(depth, merge_patch_faces, merge_label_img, merge_verts, merge_uvs, merge_faces);
    });
    printf("merge faces execution time: %lf ms\n", time * 1e3);

    return 0;
}
