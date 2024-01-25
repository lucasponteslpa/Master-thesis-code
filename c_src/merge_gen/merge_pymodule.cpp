#include <Python.h>
#include <string>
#include <cassert>
#include <cstdio>
#include <cstdlib>

#include "HalideBuffer.h"
#include "halide_malloc_trace.h"
#include "merge_mesh_verts.h"
#include "merge_mesh_faces.h"

using namespace Halide::Runtime;
using namespace Halide::Tools;


///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////


namespace PythonRuntime {


struct BuffFunc{
    int min;
    int stride;
    int extent;
};

template<typename T, int dims>
struct BuffArr{
    BuffFunc f_spec[dims];
    T* host;
};

template<typename T, int dims>
bool unpack_buffer(PyObject *py_obj,
                   int py_getbuffer_flags,
                   const char *name,
                   int dimensions,
                   Py_buffer &py_buf,
                   BuffArr<T,dims> &buff_arr,
                   bool &py_buf_valid) {
    py_buf_valid = false;

    memset(&py_buf, 0, sizeof(py_buf));
    if (PyObject_GetBuffer(py_obj, &py_buf, PyBUF_FORMAT | PyBUF_STRIDED_RO | PyBUF_ANY_CONTIGUOUS | py_getbuffer_flags) < 0) {
        PyErr_Format(PyExc_ValueError, "Invalid argument %s: Expected %d dimensions, got %d", name, dimensions, py_buf.ndim);
        return false;
    }
    py_buf_valid = true;

    if (dimensions && py_buf.ndim != dimensions) {
        PyErr_Format(PyExc_ValueError, "Invalid argument %s: Expected %d dimensions, got %d", name, dimensions, py_buf.ndim);
        return false;
    }
    /* We'll get a buffer that's either:
     * C_CONTIGUOUS (last dimension varies the fastest, i.e., has stride=1) or
     * F_CONTIGUOUS (first dimension varies the fastest, i.e., has stride=1).
     * The latter is preferred, since it's already in the format that Halide
     * needs. It can can be achieved in numpy by passing order='F' during array
     * creation. However, if we do get a C_CONTIGUOUS buffer, flip the dimensions
     * (transpose) so we can process it without having to reallocate.
     */
    int i, j, j_step;
    if (PyBuffer_IsContiguous(&py_buf, 'F')) {
        j = 0;
        j_step = 1;
    } else if (PyBuffer_IsContiguous(&py_buf, 'C')) {
        j = py_buf.ndim - 1;
        j_step = -1;
    } else {
        /* Python checks all dimensions and strides, so this typically indicates
         * a bug in the array's buffer protocol. */
        PyErr_Format(PyExc_ValueError, "Invalid buffer: neither C nor Fortran contiguous");
        return false;
    }
    for (i = 0; i < py_buf.ndim; ++i, j += j_step) {
        buff_arr.f_spec[i].min = 0;
        buff_arr.f_spec[i].stride = (int)(py_buf.strides[j] / py_buf.itemsize);  // strides is in bytes
        buff_arr.f_spec[i].extent = (int)py_buf.shape[j];
        if (py_buf.suboffsets && py_buf.suboffsets[i] >= 0) {
            // Halide doesn't support arrays of pointers. But we should never see this
            // anyway, since we specified PyBUF_STRIDED.
            PyErr_Format(PyExc_ValueError, "Invalid buffer: suboffsets not supported");
            return false;
        }
    }
    if (buff_arr.f_spec[py_buf.ndim - 1].extent * buff_arr.f_spec[py_buf.ndim - 1].stride * py_buf.itemsize != py_buf.len) {
        PyErr_Format(PyExc_ValueError, "Invalid buffer: length %ld, but computed length %ld",
                     py_buf.len, py_buf.shape[0] * py_buf.strides[0]);
        return false;
    }
    buff_arr.host = (T *)py_buf.buf;
    return true;
}

}

namespace {

template<typename T, int dimensions>
struct PyHalideBuffer {
    // Must allocate at least 1, even if d=0
    static constexpr int dims_to_allocate = (dimensions < 1) ? 1 : dimensions;

    Py_buffer py_buf;
    PythonRuntime::BuffArr<T, dimensions> buff_arr;
    bool py_buf_needs_release = false;

    bool unpack(PyObject *py_obj, int py_getbuffer_flags, const char *name) {
        return PythonRuntime::unpack_buffer(py_obj, py_getbuffer_flags, name, dimensions, py_buf, buff_arr, py_buf_needs_release);
    }

    ~PyHalideBuffer() {
        if (py_buf_needs_release) {
            PyBuffer_Release(&py_buf);
        }
    }

    PyHalideBuffer() = default;
    PyHalideBuffer(const PyHalideBuffer &other) = delete;
    PyHalideBuffer &operator=(const PyHalideBuffer &other) = delete;
    PyHalideBuffer(PyHalideBuffer &&other) = delete;
    PyHalideBuffer &operator=(PyHalideBuffer &&other) = delete;
};

}  // namespace





///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////


namespace{
    const char* const arr_kwlist[] = {
    "I_fore_idx",
    "I_back_idx",
    "I_back_labels",
    "I_fore_mask",
    "N0_idx",
    "P_out_faces",
    "P_N_faces",
    "I_out_vlabel",
    nullptr
    };

}


static PyObject* pmerge_mesh_verts(PyObject *module, PyObject *args, PyObject *kwargs){
    PyObject* py_I_fore_idx ;
    PyObject* py_I_back_idx ;
    PyObject* py_I_back_labels ;
    PyObject* py_I_fore_mask;
    int N0_vidx;
    PyObject* py_P_out_faces;
    PyObject* py_P_N_faces;
    PyObject* py_I_out_vlabel;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOOOiOOO", (char**)arr_kwlist, &py_I_fore_idx, &py_I_back_idx, &py_I_back_labels, &py_I_fore_mask, &N0_vidx, &py_P_out_faces, &py_P_N_faces, &py_I_out_vlabel)) {
        PyErr_Format(PyExc_ValueError, "Parser Args Internal error");
        return nullptr;
    }

    PyHalideBuffer<int32_t, 2> BuffIForeIdx;
    PyHalideBuffer<int32_t, 2> BuffIBackIdx;
    PyHalideBuffer<uint8_t, 2> BuffIBackLabels;
    PyHalideBuffer<uint8_t, 2> BuffBIForeMask;
    PyHalideBuffer<int32_t, 3> BuffPOutFaces;
    PyHalideBuffer<int32_t, 1> BuffPNFaces;
    PyHalideBuffer<int32_t, 2> BuffIOutVLabel;

    if (!BuffIForeIdx.unpack(py_I_fore_idx, 0, arr_kwlist[0])) return nullptr;
    if (!BuffIBackIdx.unpack(py_I_back_idx, 0, arr_kwlist[1])) return nullptr;
    if (!BuffIBackLabels.unpack(py_I_back_labels, 0, arr_kwlist[2])) return nullptr;
    if (!BuffBIForeMask.unpack(py_I_fore_mask, 0, arr_kwlist[3])) return nullptr;
    if (!BuffPOutFaces.unpack(py_P_out_faces, 0, arr_kwlist[5])) return nullptr;
    if (!BuffPNFaces.unpack(py_P_N_faces, 0, arr_kwlist[6])) return nullptr;
    if (!BuffIOutVLabel.unpack(py_I_out_vlabel, 0, arr_kwlist[7])) return nullptr;

    long int* shape_in = BuffIForeIdx.py_buf.shape;
    long int* Bshape_in = BuffBIForeMask.py_buf.shape;
    long int* PFshape_out = BuffPOutFaces.py_buf.shape;
    // long int* NFshape_out = BuffPNFaces.py_buf.shape;


    int32_t *IForeIdxHost = (int32_t*)BuffIForeIdx.buff_arr.host;
    int32_t *IBackIdxHost = (int32_t*)BuffIBackIdx.buff_arr.host;
    uint8_t *IBackLabelsHost =  (uint8_t*)BuffIBackLabels.buff_arr.host;
    uint8_t *BIForeMaskHost =  (uint8_t*)BuffBIForeMask.buff_arr.host;
    int32_t *POutFacesHost = (int32_t*)BuffPOutFaces.buff_arr.host;
    int32_t *PNFacesHost = (int32_t*)BuffPNFaces.buff_arr.host;
    int32_t *IOutVLabelHost = (int32_t*)BuffIOutVLabel.buff_arr.host;

    int w = (int)shape_in[1];
    int h = (int)shape_in[0];
    int B_w = (int)Bshape_in[1];
    int B_h = (int)Bshape_in[0];
    int N_p = (int)PFshape_out[0];
    int N_f = (int)PFshape_out[1];
    int N_c = (int)PFshape_out[2];
    int block_size=16;

    Buffer<int32_t, 2> HBuffIForeIdx = Buffer<int32_t, 2>(IForeIdxHost, w,h);
    Buffer<int32_t, 2> HBuffIBackIdx = Buffer<int32_t, 2>(IBackIdxHost, w,h);
    Buffer<uint8_t, 2> HBuffIBackLabels = Buffer<uint8_t, 2>(IBackLabelsHost, w,h);
    Buffer<uint8_t, 2> HBuffBIForeMask = Buffer<uint8_t, 2>(BIForeMaskHost, B_w,B_h);
    Buffer<int32_t, 3> HBuffPOutFaces = Buffer<int32_t, 3>(POutFacesHost, N_c, N_f, N_p);
    Buffer<int32_t, 1> HBuffPNFaces = Buffer<int32_t, 1>(PNFacesHost, N_p);
    Buffer<int32_t, 2> HBuffIOutVLabel = Buffer<int32_t, 2>(IOutVLabelHost, w,h);


    Py_BEGIN_ALLOW_THREADS
        merge_mesh_verts(HBuffIForeIdx, HBuffIBackIdx, HBuffIBackLabels, HBuffBIForeMask, N0_vidx, HBuffPOutFaces, HBuffPNFaces, HBuffIOutVLabel);
    Py_END_ALLOW_THREADS

    float result=0.0;

    return PyLong_FromLong((long)result);
}

const char* const arr_fwlist[] = {
    "I_depth",
    "P_faces",
    "I_Vidx",
    "out_verts",
    "out_uvs",
    "out_faces",
    nullptr
    };

static PyObject* pmerge_mesh_faces(PyObject *module, PyObject *args, PyObject *kwargs){
    PyObject* py_I_depth;
    PyObject* py_P_faces;
    PyObject* py_I_Vidx;
    PyObject* py_verts;
    PyObject* py_uvs;
    PyObject* py_faces;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOOOOO", (char**)arr_fwlist, &py_I_depth, &py_P_faces, &py_I_Vidx, &py_verts, &py_uvs, &py_faces)) {
        PyErr_Format(PyExc_ValueError, "Parser Args Internal error");
        return nullptr;
    }

    PyHalideBuffer<uint8_t, 2> BuffDepth;
    PyHalideBuffer<int32_t, 3> BuffPFaces;
    PyHalideBuffer<int32_t, 2> BuffIVIdx;
    PyHalideBuffer<float, 2> BuffOutputV;
    PyHalideBuffer<float, 2> BuffOutputUV;
    PyHalideBuffer<int32_t, 2> BuffOutputF;

    if (!BuffDepth.unpack(py_I_depth, 0, arr_fwlist[0])) return nullptr;
    if (!BuffPFaces.unpack(py_P_faces, 0, arr_fwlist[1])) return nullptr;
    if (!BuffIVIdx.unpack(py_I_Vidx, 0, arr_fwlist[2])) return nullptr;
    if (!BuffOutputV.unpack(py_verts, 0, arr_fwlist[3])) return nullptr;
    if (!BuffOutputUV.unpack(py_uvs, 0, arr_fwlist[4])) return nullptr;
    if (!BuffOutputF.unpack(py_faces, 0, arr_fwlist[5])) return nullptr;

    long int* shape_d = BuffDepth.py_buf.shape;
    long int* shape_f = BuffPFaces.py_buf.shape;
    long int* shape_vi = BuffIVIdx.py_buf.shape;
    long int* shape_v = BuffOutputV.py_buf.shape;
    long int* shape_uv = BuffOutputUV.py_buf.shape;
    long int* shape_out_f = BuffOutputF.py_buf.shape;

    uint8_t *depth_host =  (uint8_t*)BuffDepth.buff_arr.host;
    int32_t *faces = (int32_t*)BuffPFaces.buff_arr.host;
    int32_t *vidx = (int32_t*)BuffIVIdx.buff_arr.host;
    float *out_v = (float*)BuffOutputV.buff_arr.host;
    float *out_uv = (float*)BuffOutputUV.buff_arr.host;
    int32_t *out_f = (int32_t*)BuffOutputF.buff_arr.host;

    int w = (int)shape_d[1];
    int h = (int)shape_d[0];
    int block_size=16;
    Buffer<uint8_t> depth_hbuff= Buffer<uint8_t>(depth_host, (int)shape_d[1], (int)shape_d[0]);
    Buffer<int32_t, 3> f_hbuff= Buffer<int32_t, 3>(faces, (int)shape_f[2],(int)shape_f[1], (int)shape_f[0]);
    Buffer<int32_t, 2> vidx_hbuff= Buffer<int32_t, 2>(vidx, (int)shape_vi[1], (int)shape_vi[0]);
    Buffer<float, 2> out_v_hbuff= Buffer<float, 2>(out_v,(int)shape_v[1], (int)shape_v[0]);
    Buffer<float, 2> out_uv_hbuff= Buffer<float, 2>(out_uv,(int)shape_uv[1], (int)shape_uv[0]);
    Buffer<int32_t, 2> out_f_hbuff= Buffer<int32_t, 2>(out_f,(int)shape_out_f[1], (int)shape_out_f[0]);


    Py_BEGIN_ALLOW_THREADS
        merge_mesh_faces(depth_hbuff, f_hbuff, vidx_hbuff, out_v_hbuff, out_uv_hbuff, out_f_hbuff);
    Py_END_ALLOW_THREADS

    float result=0.0;

    return PyLong_FromLong((long)result);
}



///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////


// Exported methods are collected in a table
PyMethodDef method_table[] = {
    {"pmerge_mesh_verts", (PyCFunction) pmerge_mesh_verts, METH_VARARGS | METH_KEYWORDS, nullptr},
    {"pmerge_mesh_faces", (PyCFunction) pmerge_mesh_faces, METH_VARARGS | METH_KEYWORDS, nullptr},
    {NULL, NULL, 0, NULL} // Sentinel value ending the table
};

// A struct contains the definition of a module
PyModuleDef _module = {
    PyModuleDef_HEAD_INIT,
    "merge_mesh_verts_module", // Module name
    "This is the module docstring",
    -1,   // Optional size of the module state memory
    method_table,
    NULL, // Optional slot definitions
    NULL, // Optional traversal function
    NULL, // Optional clear function
    NULL  // Optional module deallocation function
};

// The module init function
PyMODINIT_FUNC PyInit_merge_mesh_verts_module(void) {
    return PyModule_Create(&_module);
}

