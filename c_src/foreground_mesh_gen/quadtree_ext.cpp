#include "HalideBuffer.h"
#include "quadtree.hpp"

template<typename T>
struct FuncType
{
    int32_t x_min;
    int32_t x_ext;
    int32_t y_min;
    int32_t y_ext;
    int32_t p_min;
    int32_t p_ext;

    int32_t x_stride;
    int32_t y_stride;
    int32_t p_stride;

    T* arr_p;

    FuncType(halide_buffer_t* buffer_arr){
        x_min = buffer_arr->dim[0].min;
        x_ext = buffer_arr->dim[0].extent;
        y_min = buffer_arr->dim[1].min;
        y_ext = buffer_arr->dim[1].extent;
        p_min = buffer_arr->dim[2].min;
        p_ext = buffer_arr->dim[2].extent;

        x_stride = buffer_arr->dim[0].stride;
        y_stride = buffer_arr->dim[1].stride;
        p_stride = buffer_arr->dim[2].stride;
        arr_p = ((T *)buffer_arr->host);
    }

};


extern "C"  int search_quadtree(
    halide_buffer_t * canny,
    halide_buffer_t * depth_mean,
    halide_buffer_t * b_depth_mean,
    halide_buffer_t * depth,
    halide_buffer_t * coords,
    int block_size,
    halide_buffer_t * labels,
    halide_buffer_t * mask
){
    if(canny->is_bounds_query()) {
        canny->dim[0].min = labels->dim[0].min;
        canny->dim[0].extent = labels->dim[0].extent;
        canny->dim[1].min = labels->dim[1].min;
        canny->dim[1].extent = labels->dim[1].extent;
        canny->dim[2].min = labels->dim[2].min;
        canny->dim[2].extent = labels->dim[2].extent;

    } else {
        uint8_t* canny_host_p = ((uint8_t*) canny->host);
        uint8_t* depth_host_p = ((uint8_t*) depth->host);
        int32_t* coords_host_p = ((int32_t*) coords->host);

        float* d_mean = ((float *)depth_mean->host);
        float* b_d_mean = ((float *)b_depth_mean->host);

        FuncType<int32_t> out_f = FuncType<int32_t>(labels);
        int32_t* mask_arr = ((int32_t *)mask->host);
        int32_t* faces_null=(int32_t*)nullptr;

        out_f.arr_p = out_f.arr_p - out_f.p_min*out_f.p_stride;
        mask_arr = mask_arr - mask->dim[2].min*mask->dim[2].stride;
        for (int p = out_f.p_min; p < (out_f.p_min+out_f.p_ext); p++)
        {
            int32_t ranges[10] = {
                out_f.x_min,
                block_size,
                out_f.y_min,
                block_size,
                p,
                out_f.p_min+out_f.p_ext,
                0,0,0,0
            };

            int32_t stride[6] = {
                out_f.x_stride,
                out_f.y_stride,
                out_f.p_stride,
                0,0,0
            };

            Quadtree q(
                canny_host_p,
                depth_host_p,
                d_mean,
                b_d_mean,
                coords_host_p,
                out_f.arr_p,
                mask_arr,
                faces_null,
                ranges,
                stride
            );

            for (size_t i = out_f.y_min; i < (out_f.y_min+out_f.y_ext); i++)
            {
                for (size_t j = out_f.x_min; j < (out_f.x_min+out_f.x_ext); j++)
                {
                    out_f.arr_p[p*out_f.p_stride + i*out_f.y_stride + j*out_f.x_stride] = 0;
                    mask_arr[p*out_f.p_stride + i*out_f.y_stride + j*out_f.x_stride] = 0;
                }

            }
            q.search();
        }

    }
    return 0;
}


extern "C"  int search_quadtree_faces(
    halide_buffer_t * canny,
    halide_buffer_t * coords,
    halide_buffer_t * labels,
    int block_size,
    halide_buffer_t * faces
){
    if(canny->is_bounds_query()) {
        canny->dim[0].min = labels->dim[0].min;
        canny->dim[0].extent = labels->dim[0].extent;
        canny->dim[1].min = labels->dim[1].min;
        canny->dim[1].extent = labels->dim[1].extent;
        canny->dim[2].min = labels->dim[2].min;
        canny->dim[2].extent = labels->dim[2].extent;

    } else {
        uint8_t* canny_host_p = ((uint8_t*) canny->host);
        int32_t* coords_host_p = ((int32_t*) coords->host);

        uint8_t* depth_null=(uint8_t*)nullptr;
        int32_t* mask_arr = (int32_t*)nullptr;
        float* means_null=(float*)nullptr;
        FuncType<int32_t> faces_f = FuncType<int32_t>(faces);
        FuncType<int32_t> labels_f = FuncType<int32_t>(labels);
        int32_t* faces_arr = ((int32_t *)faces->host);

        faces_arr = faces_arr - faces->dim[2].min*faces->dim[2].stride;
        for (int p = faces_f.p_min; p < (faces_f.p_min+faces_f.p_ext); p++)
        {
            int32_t ranges[10] = {
                labels_f.x_min,
                block_size,
                labels_f.y_min,
                block_size,
                p,
                faces_f.p_min+faces_f.p_ext,
                faces_f.x_min,
                faces_f.x_min+faces_f.x_ext,
                faces_f.y_min,
                faces_f.y_min+faces_f.y_ext
            };

            int32_t stride[6] = {
                labels_f.x_stride,
                labels_f.y_stride,
                labels_f.p_stride,
                faces_f.x_stride,
                faces_f.y_stride,
                faces_f.p_stride
            };

            Quadtree q(
                canny_host_p,
                depth_null,
                means_null,
                means_null,
                coords_host_p,
                labels_f.arr_p,
                mask_arr,
                faces_arr,
                ranges,
                stride
            );
            q.update_labeled();
            q.search();

            int32_t count =0;
            q.faces_search(count);
            for (size_t i = count; i < (faces_f.y_min+faces_f.y_ext); i++)
            {
                for (size_t j = faces_f.x_min; j < (faces_f.x_min+faces_f.x_ext); j++)
                {
                    faces_arr[p*faces_f.p_stride + i*faces_f.y_stride + j*faces_f.x_stride] = -1;
                }

            }
        }

    }
    return 0;
}

extern "C"  int load_Vidx(
    halide_buffer_t * I_Vlabels,
    int idx_init,
    halide_buffer_t * I_idx
){
    if(I_Vlabels->is_bounds_query()) {
        I_Vlabels->dim[0].min = I_idx->dim[0].min;
        I_Vlabels->dim[0].extent = I_idx->dim[0].extent;
        I_Vlabels->dim[1].min = I_idx->dim[1].min;
        I_Vlabels->dim[1].extent = I_idx->dim[1].extent;

    } else {
        int32_t* Vlabels_host = ((int32_t*) I_Vlabels->host);
        int32_t* I_idx_host = ((int32_t *)I_idx->host);

        int32_t x_init = I_Vlabels->dim[0].min;
        int32_t x_end = x_init + I_Vlabels->dim[0].extent;
        int32_t x_stride = I_Vlabels->dim[0].stride;
        int32_t y_init = I_Vlabels->dim[1].min;
        int32_t y_stride = I_Vlabels->dim[1].stride;
        int32_t y_end = y_init + I_Vlabels->dim[1].extent;

        I_idx_host = I_idx_host - I_idx->dim[1].min*I_idx->dim[2].stride;
        int32_t count=0;
        for (size_t y = y_init; y < y_end; y++)
        {
            for (size_t x = x_init; x < x_end; x++)
            {
                if (Vlabels_host[x*x_stride + y*y_stride]>0){
                    I_idx_host[x*x_stride + y*y_stride] = idx_init + count;
                    count++;
                }else{
                    I_idx_host[x*x_stride + y*y_stride] = -1;
                }
            }

        }

    }
    return 0;
}

extern "C"  int store_verts(
    halide_buffer_t * I_Vidx,
    halide_buffer_t * I_depth,
    int I_w,
    int I_h,
    halide_buffer_t * verts,
    halide_buffer_t * uvs
){
    if(I_Vidx->is_bounds_query()) {
        I_Vidx->dim[0].min = I_depth->dim[0].min;
        I_Vidx->dim[0].extent = I_depth->dim[0].extent;
        I_Vidx->dim[1].min = I_depth->dim[1].min;
        I_Vidx->dim[1].extent = I_depth->dim[1].extent;

    } else {
        int32_t* Vidx_host = ((int32_t*) I_Vidx->host);
        uint8_t* depth_host = ((uint8_t*) I_depth->host);
        float* verts_host = ((float *)verts->host);
        float* uvs_host = ((float *)uvs->host);

        int32_t x_init = I_Vidx->dim[0].min;
        int32_t x_end = x_init + I_Vidx->dim[0].extent;
        int32_t x_stride = I_Vidx->dim[0].stride;
        int32_t y_init = I_Vidx->dim[1].min;
        int32_t y_stride = I_Vidx->dim[1].stride;
        int32_t y_end = y_init + I_Vidx->dim[1].extent;

        verts_host = verts_host - verts->dim[1].min*verts->dim[1].stride;
        uvs_host = uvs_host - uvs->dim[1].min*uvs->dim[1].stride;
        int32_t count=0;
        for (size_t y = y_init; y < y_end; y++)
        {
            for (size_t x = x_init; x < x_end; x++)
            {
                if (Vidx_host[x*x_stride + y*y_stride]>=0){
                    int idx = Vidx_host[x*x_stride + y*y_stride];
                    verts_host[3*count] = 1.0 - (2.0*(((float)x)/((float)I_w)));
                    verts_host[3*count + 1] = 1.0 - (2.0*(((float)y)/((float)I_h)));
                    verts_host[3*count + 2] = ((float)depth_host[x*x_stride + y*y_stride])/255.0;

                    uvs_host[3*count] = ((((float)x)/((float)I_w)));
                    uvs_host[3*count + 1] = 1.0 - ((((float)y)/((float)I_h)));
                    uvs_host[3*count + 2] = -1;
                    count++;
                }
            }

        }

    }
    return 0;
}

extern "C"  int store_faces(
    halide_buffer_t * P_faces,
    halide_buffer_t * I_Vidx,
    int I_w,
    int I_h,
    halide_buffer_t * faces
){
    if(P_faces->is_bounds_query()) {

    } else {
        int32_t* P_faces_host = ((int32_t*) P_faces->host);
        int32_t* I_Vidx_host = ((int32_t*) I_Vidx->host);
        int32_t* faces_host = ((int32_t *)faces->host);

        int32_t c_init = P_faces->dim[0].min;
        int32_t c_end = c_init + P_faces->dim[0].extent;
        int32_t c_stride = P_faces->dim[0].stride;
        int32_t f_init = P_faces->dim[1].min;
        int32_t f_end = f_init + P_faces->dim[1].extent;
        int32_t f_stride = P_faces->dim[1].stride;
        int32_t p_init = P_faces->dim[2].min;
        int32_t p_end = p_init + P_faces->dim[2].extent;
        int32_t p_stride = P_faces->dim[2].stride;

        faces_host = faces_host - faces->dim[1].min*faces->dim[1].stride;
        int32_t count=0;
        int last=0;
        for (size_t p = p_init; p < p_end; p++)
        {
            for (size_t f = f_init; f < f_end; f++)
            {
                if (P_faces_host[f*f_stride + p*p_stride]>=0){
                    for (size_t c = 0; c < 3; c++)
                    {
                        int coord = P_faces_host[f*f_stride + p*p_stride + c];
                        if (coord>=0 && coord<(I_w*I_h)){
                            faces_host[3*count+c] = I_Vidx_host[coord];
                            last = coord;

                        }else{
                            faces_host[3*count+c] = last;
                        }
                    }
                    if ((count+1)<((faces->dim[0].min + faces->dim[0].extent) + (faces->dim[1].min + faces->dim[1].extent)*faces->dim[1].stride))
                        count++;
                }
            }

        }

    }
    return 0;
}