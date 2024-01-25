#include "HalideBuffer.h"

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

    T* host;

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
        host = ((T *)buffer_arr->host);
    }

};


extern "C"  int search_mid_faces(
    halide_buffer_t * PIForeIdx,
    halide_buffer_t * PIBackIdx,
    halide_buffer_t * PIBackLabels,
    halide_buffer_t * PForeMask,
    halide_buffer_t * PVIdx,
    int block_size,
    halide_buffer_t * P_faces
){
    if(PIForeIdx->is_bounds_query()) {


    } else {
        FuncType<int32_t> f_idx = FuncType<int32_t>(PIForeIdx);
        FuncType<int32_t> b_idx = FuncType<int32_t>(PIBackIdx);
        FuncType<uint8_t> b_l = FuncType<uint8_t>(PIBackLabels);
        uint8_t* f_m_host = ((uint8_t*) PForeMask->host);
        int32_t* p_vidx_host = ((int32_t*) PVIdx->host);

        int32_t* P_faces_host = ((int32_t*) P_faces->host);

        int32_t c_init = P_faces->dim[0].min;
        int32_t c_end = c_init + P_faces->dim[0].extent;
        int32_t c_stride = P_faces->dim[0].stride;
        int32_t f_init = P_faces->dim[1].min;
        int32_t f_end = f_init + P_faces->dim[1].extent;
        int32_t f_stride = P_faces->dim[1].stride;
        int32_t p_init = P_faces->dim[2].min;
        int32_t p_end = p_init + P_faces->dim[2].extent;
        int32_t p_stride = P_faces->dim[2].stride;

        P_faces_host = P_faces_host - p_init*p_stride;
        int i_init=0, j_init=0, i_end=block_size, j_end=block_size;
        int count = 0;
        int32_t i_s = -1, j_s = -1;
        int32_t f_idx_pix = 0, b_idx_pix=0, b_l_pix = 0;
        int32_t f_idx_pix_s = 0, b_idx_pix_s=0, b_l_pix_s = 0;
        int mid_x = block_size/2, mid_y = block_size/2;
        for (int p = p_init; p < (p_end); p++)
        {
            count = 0;
            if (f_m_host[p]==2){
                b_idx_pix =   b_idx.host[p*b_idx.p_stride + i_init*b_idx.x_stride + j_init*b_idx.y_stride];
                P_faces_host[p*p_stride + 0*c_stride + count*f_stride] = b_idx_pix;
                b_idx_pix =   b_idx.host[p*b_idx.p_stride + i_end*b_idx.x_stride + j_init*b_idx.y_stride];
                P_faces_host[p*p_stride + 1*c_stride + count*f_stride] = b_idx_pix;
                b_idx_pix =   b_idx.host[p*b_idx.p_stride + i_init*b_idx.x_stride + j_end*b_idx.y_stride];
                P_faces_host[p*p_stride + 2*c_stride + count*f_stride] = b_idx_pix;

                b_idx_pix =   b_idx.host[p*b_idx.p_stride + i_end*b_idx.x_stride + j_init*b_idx.y_stride];
                P_faces_host[p*p_stride + 0*c_stride + (count+1)*f_stride] = b_idx_pix;
                b_idx_pix =   b_idx.host[p*b_idx.p_stride + i_init*b_idx.x_stride + j_end*b_idx.y_stride];
                P_faces_host[p*p_stride + 1*c_stride + (count+1)*f_stride] = b_idx_pix;
                b_idx_pix =   b_idx.host[p*b_idx.p_stride + i_end*b_idx.x_stride + j_end*b_idx.y_stride];
                P_faces_host[p*p_stride + 2*c_stride + (count+1)*f_stride] = b_idx_pix;
                count += 2;
                for (size_t f = count; f < f_end; f++)
                {
                    for (size_t c = 0; c < c_end; c++)
                    {
                        P_faces_host[p*p_stride + c*c_stride + f*f_stride] = -1;
                    }

                }
            }
            else if (f_m_host[p]==1){
                // P_faces_host[p*p_stride + 0*c_stride + count*f_stride] = 2;
                i_s = -1, j_s = -1;
                for (int i = i_init; i <= i_end; i++)
                {
                    f_idx_pix =   f_idx.host[p*f_idx.p_stride + i*f_idx.x_stride + j_init*f_idx.y_stride];
                    f_idx_pix_s = f_idx.host[p*f_idx.p_stride + i_s*f_idx.x_stride + j_init*f_idx.y_stride];
                    b_idx_pix =   b_idx.host[p*b_idx.p_stride + i*b_idx.x_stride + j_init*b_idx.y_stride];
                    b_idx_pix_s = b_idx.host[p*b_idx.p_stride + i_s*b_idx.x_stride + j_init*b_idx.y_stride];
                    b_l_pix =     b_l.host[p*b_l.p_stride + i*b_l.x_stride + j_init*b_l.y_stride];
                    b_l_pix_s =   b_l.host[p*b_l.p_stride + i_s*b_l.x_stride + j_init*b_l.y_stride];

                    if (f_idx_pix>0 || b_l_pix>0){
                        if (i!=i_s && i_s!=-1){
                            P_faces_host[p*p_stride + 0*c_stride + count*f_stride] = f_idx_pix>0?f_idx_pix:b_idx_pix;
                            P_faces_host[p*p_stride + 1*c_stride + count*f_stride] = f_idx_pix_s>0?f_idx_pix_s:b_idx_pix_s;
                            P_faces_host[p*p_stride + 2*c_stride + count*f_stride] = p_vidx_host[p];
                            count = count + 1;
                            i_s = i;
                        }
                        else if(i_s==-1){
                            i_s =i;
                        }
                    }

                }

                i_s = -1, j_s = -1;
                for (int i = i_init; i <= i_end; i++)
                {
                    f_idx_pix =   f_idx.host[p*f_idx.p_stride + i*f_idx.x_stride + j_end*f_idx.y_stride];
                    f_idx_pix_s = f_idx.host[p*f_idx.p_stride + i_s*f_idx.x_stride + j_end*f_idx.y_stride];
                    b_idx_pix =   b_idx.host[p*b_idx.p_stride + i*b_idx.x_stride + j_end*b_idx.y_stride];
                    b_idx_pix_s = b_idx.host[p*b_idx.p_stride + i_s*b_idx.x_stride + j_end*b_idx.y_stride];
                    b_l_pix =     b_l.host[p*b_l.p_stride + i*b_l.x_stride + j_end*b_l.y_stride];
                    b_l_pix_s =   b_l.host[p*b_l.p_stride + i_s*b_l.x_stride + j_end*b_l.y_stride];

                    if (f_idx_pix>0 || b_l_pix>0){
                        if (i!=i_s && i_s!=-1){
                            P_faces_host[p*p_stride + 0*c_stride + count*f_stride] = f_idx_pix>0?f_idx_pix:b_idx_pix;
                            P_faces_host[p*p_stride + 1*c_stride + count*f_stride] = f_idx_pix_s>0?f_idx_pix_s:b_idx_pix_s;
                            P_faces_host[p*p_stride + 2*c_stride + count*f_stride] = p_vidx_host[p];
                            count = count + 1;
                            i_s = i;
                        }
                        else if(i_s==-1){
                            i_s =i;
                        }
                    }

                }

                i_s = -1, j_s = -1;
                for (int j = i_init; j <= i_end; j++)
                {
                    f_idx_pix =   f_idx.host[p*f_idx.p_stride + j*f_idx.y_stride + i_init*f_idx.x_stride];
                    f_idx_pix_s = f_idx.host[p*f_idx.p_stride + j_s*f_idx.y_stride + i_init*f_idx.x_stride];
                    b_idx_pix =   b_idx.host[p*b_idx.p_stride + j*b_idx.y_stride + i_init*b_idx.x_stride];
                    b_idx_pix_s = b_idx.host[p*b_idx.p_stride + j_s*b_idx.y_stride + i_init*b_idx.x_stride];
                    b_l_pix =     b_l.host[p*b_l.p_stride + j*b_l.y_stride + i_init*b_l.x_stride];
                    b_l_pix_s =   b_l.host[p*b_l.p_stride + j_s*b_l.y_stride + i_init*b_l.x_stride];

                    if (f_idx_pix>0 || b_l_pix>0){
                        if (j!=j_s && j_s!=-1){
                            P_faces_host[p*p_stride + 0*c_stride + count*f_stride] = f_idx_pix>0?f_idx_pix:b_idx_pix;
                            P_faces_host[p*p_stride + 1*c_stride + count*f_stride] = f_idx_pix_s>0?f_idx_pix_s:b_idx_pix_s;
                            P_faces_host[p*p_stride + 2*c_stride + count*f_stride] = p_vidx_host[p];
                            count = count + 1;
                            j_s = j;
                        }
                        else if(j_s==-1){
                            j_s =j;
                        }
                    }

                }

                i_s = -1, j_s = -1;
                for (int j = i_init; j <= i_end; j++)
                {
                    f_idx_pix =   f_idx.host[p*f_idx.p_stride + j*f_idx.y_stride + i_end*f_idx.x_stride];
                    f_idx_pix_s = f_idx.host[p*f_idx.p_stride + j_s*f_idx.y_stride + i_end*f_idx.x_stride];
                    b_idx_pix =   b_idx.host[p*b_idx.p_stride + j*b_idx.y_stride + i_end*b_idx.x_stride];
                    b_idx_pix_s = b_idx.host[p*b_idx.p_stride + j_s*b_idx.y_stride + i_end*b_idx.x_stride];
                    b_l_pix =     b_l.host[p*b_l.p_stride + j*b_l.y_stride + i_end*b_l.x_stride];
                    b_l_pix_s =   b_l.host[p*b_l.p_stride + j_s*b_l.y_stride + i_end*b_l.x_stride];

                    if (f_idx_pix>0 || b_l_pix>0){
                        if (j!=j_s && j_s!=-1){
                            P_faces_host[p*p_stride + 0*c_stride + count*f_stride] = f_idx_pix>0?f_idx_pix:b_idx_pix;
                            P_faces_host[p*p_stride + 1*c_stride + count*f_stride] = f_idx_pix_s>0?f_idx_pix_s:b_idx_pix_s;
                            P_faces_host[p*p_stride + 2*c_stride + count*f_stride] = p_vidx_host[p];

                            count = count + 1;
                            j_s = j;
                        }
                        else if(j_s==-1){
                            j_s =j;
                        }
                    }

                }
                for (size_t f = count; f < f_end; f++)
                {
                    for (size_t c = 0; c < c_end; c++)
                    {
                        P_faces_host[p*p_stride + c*c_stride + f*f_stride] = -1;
                    }

                }



            }else{
                // P_faces_host[p*p_stride + 0*c_stride + count*f_stride] = 1;
                for (size_t f = f_init; f < f_end; f++)
                {
                    for (size_t c = 0; c < c_end; c++)
                    {
                        P_faces_host[p*p_stride + c*c_stride + f*f_stride] = -1;
                    }

                }
            }
        }

    }
    return 0;
}

extern "C"  int load_vert_idx(
    halide_buffer_t * PForeMask,
    int idx_init,
    halide_buffer_t * PVertIdx
){
    if(PForeMask->is_bounds_query()) {
        PForeMask->dim[0].min = PVertIdx->dim[0].min;
        PForeMask->dim[0].extent = PVertIdx->dim[0].extent;

    } else {
        uint8_t* PForeMask_host = ((uint8_t*) PForeMask->host);
        int32_t* PVertIdx_host = ((int32_t *)PVertIdx->host);

        int32_t p_init = PVertIdx->dim[0].min;
        int32_t p_end = p_init + PVertIdx->dim[0].extent;

        PVertIdx_host = PVertIdx_host - PVertIdx->dim[0].min*PVertIdx->dim[0].stride;
        int32_t count=0;
        for (size_t p = p_init; p < p_end; p++)
        {
            if (PForeMask_host[p]==1){
                PVertIdx_host[p] = idx_init + count;
                count++;
            }else{
                PVertIdx_host[p] = -1;
            }
        }

    }
    return 0;
}


extern "C"  int merge_store_verts(
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
                int coord = Vidx_host[x*x_stride + y*y_stride];
                if (coord>=0){
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

extern "C"  int merge_store_faces(
    halide_buffer_t * P_faces,
    int I_w,
    int I_h,
    halide_buffer_t * faces
){
    if(P_faces->is_bounds_query()) {

    } else {
        int32_t* P_faces_host = ((int32_t*) P_faces->host);
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
        for (size_t p = p_init; p < p_end; p++)
        {
            for (size_t f = f_init; f < f_end; f++)
            {
                if (P_faces_host[p*p_stride]>=0){
                    for (size_t c = 0; c < 3; c++)
                    {
                        int coord = P_faces_host[f*f_stride + p*p_stride + c];
                        if (P_faces_host[f*f_stride + p*p_stride] >= 0){
                            if (count<(faces->dim[1].min+faces->dim[1].extent))
                                faces_host[3*count+c] = coord;
                        }
                        else break;
                    }
                }
                if (P_faces_host[f*f_stride + p*p_stride] < 0) break;
                count++;
            }

        }

    }
    return 0;
}