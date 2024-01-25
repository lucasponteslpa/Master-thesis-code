#ifndef _QUADTREE_
#define _QUADTREE_

#include <stdlib.h>
#include <cmath>
#include <string>
using namespace std;

class Quadtree{
    public:
        int32_t i_init, i_end, j_init, j_end;
        int32_t p, p_ext, p_stride, p_m;
        int32_t f0_stride, f1_stride, pf_stride;
        int32_t fx_min, fy_min, fx_end, fy_end;
        int32_t max_points;
        int32_t t_depth;
        char node_type[10];
        int32_t max_dim;
        int32_t count;
        int32_t mid_coord[3];
        int32_t mid_label;
        int32_t img_dims[2];
        bool divided;
        bool labeled;

        int32_t d0_stride;
        int32_t d1_stride;

        Quadtree* nw;
        Quadtree* ne;
        Quadtree* sw;
        Quadtree* se;

        uint8_t* canny;
        uint8_t* depth;
        float* depth_mean;
        float* b_depth_mean;
        int32_t* coords;
        int32_t* init_faces;

        int32_t* labels;
        int32_t* mask;
        int32_t* faces;

        Quadtree(
            uint8_t* canny_in,
            uint8_t* depth_in,
            float* depth_mean_in,
            float* b_depth_mean_in,
            int32_t* coords_in,
            int32_t* out_l,
            int32_t* mask_out,
            int32_t* faces_out,
            int32_t ranges[10],
            int32_t strides[6]
        ){
            canny = canny_in;
            depth = depth_in;
            depth_mean = depth_mean_in;
            b_depth_mean = b_depth_mean_in;
            coords = coords_in;
            labels = out_l;
            mask = mask_out;
            faces = faces_out;
            mid_label = 0;

            i_init=ranges[0], i_end=ranges[1], j_init=ranges[2], j_end=ranges[3];
            p = ranges[4], p_ext=ranges[5], p_stride = strides[2];

            f0_stride=strides[3], f1_stride=strides[4], pf_stride=strides[5];

            d0_stride = strides[0];
            d1_stride = strides[1];

            init_mid_coords(mid_coord);

            max_points = 1;
            t_depth = 0;
            char t[10] = "root";
            strcpy(node_type,t);
            max_dim = 0;
            count = 0;
            divided = false;
            labeled = false;
        }

        Quadtree(
            uint8_t* canny_in,
            uint8_t* depth_in,
            float_t* depth_mean_in,
            float_t* b_depth_mean_in,
            int32_t* coords_in,
            int32_t* out_l,
            int32_t* mask_out,
            int32_t* faces_out,
            int32_t ranges[10],
            int32_t strides[6],
            int32_t d,
            char name[10]
        ){
            canny = canny_in;
            depth = depth_in;
            depth_mean = depth_mean_in;
            b_depth_mean = b_depth_mean_in;
            coords = coords_in;
            labels = out_l;
            mask = mask_out;
            faces = faces_out;
            i_init=ranges[0], i_end=ranges[1], j_init=ranges[2], j_end=ranges[3];
            p = ranges[4], p_ext=ranges[5], p_stride = strides[2];

            f0_stride=strides[3], f1_stride=strides[4], pf_stride=strides[5];

            d0_stride = strides[0];
            d1_stride = strides[1];

            init_mid_coords(mid_coord);

            max_points = 1;
            t_depth = d;
            strcpy(node_type,name);
            max_dim = 0;
            count = 0;
            divided = false;
            labeled = false;
        }

        int divide()
        {
            divided = true;
            int32_t i_stride = static_cast<int32_t>(floor((i_end-i_init)/2));
            int32_t j_stride = static_cast<int32_t>(floor((j_end-j_init)/2));

            int32_t sw_range[6] = {i_init, i_init+(i_stride), j_init, j_init+(j_stride), p, p_stride};
            int32_t se_range[6] = {i_init, i_init+(i_stride), j_end-(j_stride), j_end  , p, p_stride};
            int32_t nw_range[6] = {i_end-(i_stride), i_end, j_init, j_init+(j_stride)  , p, p_stride};
            int32_t ne_range[6] = {i_end-(i_stride), i_end, j_end-(j_stride), j_end    , p, p_stride};

            char name_nw[10] = "nw";
            char name_ne[10] = "ne";
            char name_sw[10] = "sw";
            char name_se[10] = "se";

            int32_t strides[6] = {d0_stride, d1_stride, p_stride, f0_stride, f1_stride, pf_stride};
            nw = new Quadtree(canny, depth, depth_mean, b_depth_mean, coords, labels, mask, faces, nw_range, strides, t_depth+1, name_nw);
            ne = new Quadtree(canny, depth, depth_mean, b_depth_mean, coords, labels, mask, faces, ne_range, strides, t_depth+1, name_ne);
            sw = new Quadtree(canny, depth, depth_mean, b_depth_mean, coords, labels, mask, faces, sw_range, strides, t_depth+1, name_sw);
            se = new Quadtree(canny, depth, depth_mean, b_depth_mean, coords, labels, mask, faces, se_range, strides, t_depth+1, name_se);
            if (labeled){
                nw->update_labeled();
                ne->update_labeled();
                sw->update_labeled();
                se->update_labeled();
            }
            return 0;
        }

        int search()
        {
            int f_count = 0;
            if (!labeled){
                f_count = label_corners();
            };
            if (check_canny_region()){
                if((abs(i_init-i_end)>1) && (abs(j_init - j_end)>1)){
                    divide();
                    nw->search();
                    ne->search();
                    sw->search();
                    se->search();
                }
            }else{
                if(!labeled){
                    if((abs(i_init-i_end)>1) && (abs(j_init - j_end)>1)){
                        mid_label = label_mid_coords();
                        mask_region();
                    }
                }
            }
            return 0;
        }

        int faces_search(int &f_count)
        {

            uint8_t count_v[4] = {0,0,0,0};

            if((abs(i_init-i_end)>1) && (abs(j_init - j_end)>1)){
                if (divided){
                    nw->faces_search(f_count);
                    ne->faces_search(f_count);
                    sw->faces_search(f_count);
                    se->faces_search(f_count);
                }else{
                    check_index_region(count_v);
                    uint8_t check_corners=0;
                    for (size_t i = 0; i < 4; i++)
                    {
                        if (count_v[i]>1)check_corners++;
                    }

                    if (check_corners>1 && t_depth>0){
                        for (size_t i = 0; i < 4; i++)
                        {
                            if ( labels[p*p_stride + mid_coord[0]*d0_stride + mid_coord[1]*d1_stride]==1){
                                if (count_v[i]>1){
                                    get_faces_region(i, f_count);
                                }
                            }
                        }
                    }

                }
            }else if((abs(i_init-i_end)==1) && (abs(j_init - j_end)==1)){
                int32_t corners_idx[4] = {
                    p*p_stride + i_init*d0_stride + j_init*d1_stride,
                    p*p_stride + i_end*d0_stride + j_init*d1_stride,
                    p*p_stride + i_init*d0_stride + j_end*d1_stride,
                    p*p_stride + i_end*d0_stride + j_end*d1_stride
                };
                int c = 0, idx=0;
                int corners_coords[4] = {-1,-1,-1,-1};
                for (size_t i = 0; i < 4; i++)
                {
                    idx = corners_idx[i];
                    if (labels[idx]>0) {
                        corners_coords[c] = coords[idx];
                        if (c>=2){
                            faces[p*pf_stride + 0*f0_stride + f_count*f1_stride] = corners_coords[c-2];
                            faces[p*pf_stride + 1*f0_stride + f_count*f1_stride] = corners_coords[c-1];
                            faces[p*pf_stride + 2*f0_stride + f_count*f1_stride] = corners_coords[c];
                            f_count = f_count + 1;
                        }
                        c++;
                    }
                }


            }
            return 0;
        }

        void update_labeled(){
            labeled = labeled?false:true;
        }

        void update_img_dims(int width, int height){
            img_dims[0] = width;
            img_dims[1] = height;
        }

    private:
        void init_mid_coords(int32_t * m_coord)
        {
            int32_t i_stride = static_cast<int32_t>(floor((i_end-i_init)/2));
            int32_t j_stride = static_cast<int32_t>(floor((j_end-j_init)/2));
            m_coord[0] = i_end - i_stride;
            m_coord[1] = j_end - j_stride;
            m_coord[2] = -1;
        }

        void initialize_edges()
        {
            for (int i = fx_min; i <= fx_end; i++)
            {
                for (int j = fy_min; j <= fy_end; j++)
                {
                    faces[p*pf_stride + i*f0_stride + j*f1_stride] = -1;
                }

            }
        }
        bool check_canny_region()
        {
            for (int i = i_init; i <= i_end; i++)
            {
                for (int j = j_init; j <= j_end; j++)
                {
                    if (canny[p*p_stride + i*d0_stride + j*d1_stride]==255) return true;
                    // return true;
                }

            }
            return false;

        }

        uint8_t* check_index_region(uint8_t* count_v)
        {
            for (int i = i_init; i <= i_end; i++)
            {
                if (labels[p*p_stride + i*d0_stride + j_init*d1_stride]==1) count_v[0]++;

            }
            for (int j = j_init; j <= j_end; j++)
            {
                if (labels[p*p_stride + i_end*d0_stride + j*d1_stride]==1) count_v[1]++;
            }
            for (int i = i_init; i <= i_end; i++)
            {
                if (labels[p*p_stride + i*d0_stride + j_end*d1_stride]==1) count_v[2]++;

            }
            for (int j = j_init; j <= j_end; j++)
            {
                if (labels[p*p_stride + i_init*d0_stride + j*d1_stride]==1) count_v[3]++;
            }
            return count_v;

        }

        void get_faces_region(uint8_t edge_idx, int32_t &count)
        {
            int32_t i_s = -1;
            int32_t j_s = -1;
            int32_t mid_x = mid_coord[0], mid_y=mid_coord[1];
            switch (edge_idx)
            {

            case 0:
                /* code */
                for (int i = i_init; i <= i_end; i++)
                {
                    if (labels[p*p_stride + i*d0_stride + j_init*d1_stride]==1){
                        if (i!=i_s && i_s!=-1){
                            faces[p*pf_stride + 0*f0_stride + count*f1_stride] = coords[p*p_stride + i_s*d0_stride + j_init*d1_stride];
                            faces[p*pf_stride + 1*f0_stride + count*f1_stride] = coords[p*p_stride + i*d0_stride + j_init*d1_stride];
                            faces[p*pf_stride + 2*f0_stride + count*f1_stride] = coords[p*p_stride + mid_x*d0_stride + mid_y*d1_stride];
                            count = count + 1;
                            i_s = i;
                        }
                        else if(i_s==-1){
                            i_s =i;
                        }
                    };

                }
                break;

            case 1:
                for (int j = j_init; j <= j_end; j++)
                {
                    if (labels[p*p_stride + i_end*d0_stride + j*d1_stride]==1){
                        if (j!=j_s && j_s!=-1){
                            faces[p*pf_stride + 0*f0_stride + count*f1_stride] = coords[p*p_stride + i_end*d0_stride + j_s*d1_stride];
                            faces[p*pf_stride + 1*f0_stride + count*f1_stride] = coords[p*p_stride + i_end*d0_stride + j*d1_stride];
                            faces[p*pf_stride + 2*f0_stride + count*f1_stride] = coords[p*p_stride + mid_x*d0_stride + mid_y*d1_stride];
                            count = count + 1;
                            j_s = j;
                        }
                        else if(j_s==-1){
                            j_s =j;
                        }
                    }
                }
                break;

            case 2:
                for (int i = i_init; i <= i_end; i++)
                {
                    if (labels[p*p_stride + i*d0_stride + j_end*d1_stride]==1){
                        if (i!=i_s && i_s!=-1){
                            faces[p*pf_stride + 0*f0_stride + count*f1_stride] = coords[p*p_stride + i_s*d0_stride + j_end*d1_stride];
                            faces[p*pf_stride + 1*f0_stride + count*f1_stride] = coords[p*p_stride + i*d0_stride + j_end*d1_stride];
                            faces[p*pf_stride + 2*f0_stride + count*f1_stride] = coords[p*p_stride + mid_x*d0_stride + mid_y*d1_stride];
                            count = count + 1;
                            i_s = i;
                        }
                        else if(i_s==-1){
                            i_s =i;
                        }
                    };

                }
                break;

            default:
                for (int j = j_init; j <= j_end; j++)
                {
                    if (labels[p*p_stride + i_init*d0_stride + j*d1_stride]==1){
                        if (j!=j_s && j_s!=-1){
                            faces[p*pf_stride + 0*f0_stride + count*f1_stride] = coords[p*p_stride + i_init*d0_stride + j_s*d1_stride];
                            faces[p*pf_stride + 1*f0_stride + count*f1_stride] = coords[p*p_stride + i_init*d0_stride + j*d1_stride];
                            faces[p*pf_stride + 2*f0_stride + count*f1_stride] = coords[p*p_stride + mid_x*d0_stride + mid_y*d1_stride];
                            count = count + 1;
                            j_s = j;
                        }
                        else if(j_s==-1){
                            j_s =j;
                        }
                    }
                }
                break;
            }

        }

        int label_mid_coords()
        {
            int32_t i = mid_coord[0];
            int32_t j = mid_coord[1];
            float fore_diff=2;
            float back_diff=2;
            fore_diff = mid_fore_diff();
            back_diff = mid_back_diff();
            if (fore_diff >= back_diff) labels[p*p_stride + i*d0_stride + j*d1_stride] = 0;
            else labels[p*p_stride + i*d0_stride + j*d1_stride] = 1;
            return 0;
        }

        int label_corners()
        {
            float fore_diff=0, back_diff=0;
            int count=0;
            for (size_t i = 0; i < 4; i++)
            {
                fore_diff = depth_fore_diff(i);
                back_diff = depth_back_diff(i);
                if (fore_diff >= back_diff) label(i, 0);
                else {
                    label(i, 1);
                    count++;
                }
            }
            if (count<=2) return 0;
            else return 1;
        }

        float mid_fore_diff(){
            int i = mid_coord[0];
            int j = mid_coord[1];
            float diff = (float)(depth[p*p_stride + i*d0_stride + j*d1_stride]) - depth_mean[p];
            return abs(diff);
        }

        float mid_back_diff(){
            int i = mid_coord[0];
            int j = mid_coord[1];
            float diff = (float)(depth[p*p_stride + i*d0_stride + j*d1_stride]) - b_depth_mean[p];
            return abs(diff);
        }

        float depth_fore_diff(int corner_idx){
            int i = corner_idx%2==0?i_init:i_end;
            int j = corner_idx<2?j_init:j_end;
            float diff = (float)(depth[p*p_stride + i*d0_stride + j*d1_stride]) - depth_mean[p];
            return abs(diff);
        }

        float depth_back_diff(int corner_idx){
            int i = corner_idx%2==0?i_init:i_end;
            int j = corner_idx<2?j_init:j_end;
            float diff = (float)(depth[p*p_stride + i*d0_stride + j*d1_stride]) - b_depth_mean[p];
            return abs(diff);
        }

        void label(int corner_idx, int label){
            int i = corner_idx%2==0?i_init:i_end;
            int j = corner_idx<2?j_init:j_end;
            labels[p*p_stride + i*d0_stride + j*d1_stride] = label;
            mask[p*p_stride + i*d0_stride + j*d1_stride] = label;
            // v_depth[p*p_stride + i*d0_stride + j*d1_stride] = (int32_t)depth[p*p_stride + i*d0_stride + j*d1_stride];
        }

        void mask_region(){
            const int l0 = labels[p*p_stride + i_init*d0_stride + j_init*d1_stride];
            const int l1 = labels[p*p_stride + i_end*d0_stride + j_init*d1_stride];
            const int l2 = labels[p*p_stride + i_init*d0_stride + j_end*d1_stride];
            const int l3 = labels[p*p_stride + i_end*d0_stride + j_end*d1_stride];

            int x_init = i_init;
            int x_end = i_end;
            const int count_l = (l0>0?1:0) + (l1>0?1:0) + (l2>0?1:0) + (l3>0?1:0);
            if (count_l>3){
                for (size_t y = j_init; y < j_end; y++)
                {
                    x_init = (l0>0?(l2>0?i_init:(i_init+(y-j_init))):(i_end-(y-j_init)));
                    x_end = (l1>0?(l3>0?i_end:(i_end-(y-j_init))):(y-j_init));
                    for (size_t x = x_init; x < x_end; x++)
                    {
                        mask[p*p_stride + x*d0_stride + y*d1_stride] = 1;
                    }

                }
            }

        }


};

#endif
