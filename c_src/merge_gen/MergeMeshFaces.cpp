#include "Halide.h"
#include <stdio.h>
namespace {
using namespace Halide;
using namespace Halide::ConciseCasts;
using namespace Halide::Internal;

class MergeMeshFaces : public Halide::Generator<MergeMeshFaces> {
public:

    GeneratorParam<int> block_size{"block_size",16};

    Input<Buffer<uint8_t, 2>> I_depth{"I_depth"};
    Input<Buffer<int32_t, 3>> P_faces{"P_faces"};
    Input<Buffer<int32_t, 2>> I_Vidx{"I_Vidx"};

    Output<Buffer<float, 2>> out_verts{"out_verts"};
    Output<Buffer<float, 2>> out_uvs{"out_uvs"};
    Output<Buffer<int32_t, 2>> out_faces{"out_faces"};

    void generate() {
        Expr w = I_depth.dim(0).extent();
        Expr h = I_depth.dim(1).extent();
        Expr P = P_faces.dim(2).extent();

        vert_arr.define_extern(
            "merge_store_verts",
            {
                Func(I_Vidx),
                Func(I_depth),
                w,
                h
            },
            {Float(32), Float(32)},
            {c, v},
            NameMangling::C
        );
        Expr v_proxy_expr = I_Vidx(0, 0) + I_Vidx(w-1, h-1) + I_depth(0, 0) + I_depth(w-1, h-1);
        vert_arr.function().extern_definition_proxy_expr() = (v_proxy_expr, v_proxy_expr);

        face_arr.define_extern(
            "merge_store_faces",
            {
                Func(P_faces),
                w,
                h
            },
            {Int(32)},
            {c, f},
            NameMangling::C
        );
        Expr f_proxy_expr = P_faces(0, 0, 0) + P_faces(2, 2*(block_size+1)*(block_size+1), P-1);
        face_arr.function().extern_definition_proxy_expr() = f_proxy_expr;

        out_faces(c,f) = face_arr(c,f);
        out_verts(c,v) = vert_arr(c,v)[0];
        out_uvs(c,v) = vert_arr(c,v)[1];
    }

    void schedule() {
        int vector_size = natural_vector_size(UInt(8));

        if(using_autoscheduler()) {
            I_depth.set_estimates({{0, 656},{0, 656}});
        } else {
            Var po,pi;
            int split_size = 32;
            vert_arr.compute_root();
            out_verts.compute_root();
            out_uvs.compute_root();
            face_arr.compute_root();
            out_faces.compute_root();
        }

    }

private:
    Var x{"x"}, y{"y"}, x_p{"x_p"}, y_p{"y_p"}, p{"p"}, v{"v"}, c{"c"},f{"f"};
    Func  depth_bound{"depth_bound"},labels_bound{"labels_bound"},I_vert_idx{"I_vert_idx"},
    vert_arr{"vert_arr"}, face_arr{"face_arr"};
    RDom r_p, rm_p, r_f;
};

}  // namespace

HALIDE_REGISTER_GENERATOR(MergeMeshFaces, merge_mesh_faces)
