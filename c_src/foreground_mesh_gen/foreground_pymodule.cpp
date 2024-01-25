#include <Python.h>
#include <string>
#include <cassert>
#include <cstdio>
#include <cstdlib>

#include "HalideBuffer.h"
#include "halide_malloc_trace.h"
#include "foreground_mesh_verts.h"
#include "foreground_mesh_faces.h"

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
    "canny",
    "back_canny",
    "depth",
    // "out_l",
    // "out_d",
    "out_f",
    "out_Nf",
    "out_limg",
    "out_mimg",
    nullptr
    };

}


static PyObject* pforeground_mesh_verts(PyObject *module, PyObject *args, PyObject *kwargs){
    PyObject* py_canny;
    PyObject* py_back_canny;
    PyObject* py_depth;
    // PyObject* py_out_l;
    // PyObject* py_out_d;
    PyObject* py_out_f;
    PyObject* py_out_Nf;
    PyObject* py_out_limg;
    PyObject* py_out_mimg;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOOOOOO", (char**)arr_kwlist, &py_canny, &py_back_canny, &py_depth, &py_out_f, &py_out_Nf, &py_out_limg, &py_out_mimg)) {
        PyErr_Format(PyExc_ValueError, "Parser Args Internal error");
        return nullptr;
    }

    PyHalideBuffer<uint8_t, 2> BuffCanny;
    PyHalideBuffer<uint8_t, 2> BuffBackCanny;
    PyHalideBuffer<uint8_t, 2> BuffDepth;
    // PyHalideBuffer<int32_t, 3> BuffOutputL;
    // PyHalideBuffer<int32_t, 3> BuffOutputD;
    PyHalideBuffer<int32_t, 3> BuffOutputF;
    PyHalideBuffer<int32_t, 1> BuffOutputNF;
    PyHalideBuffer<int32_t, 2> BuffOutputLImg;
    PyHalideBuffer<int32_t, 2> BuffOutputMImg;

    if (!BuffCanny.unpack(py_canny, 0, arr_kwlist[0])) return nullptr;
    if (!BuffBackCanny.unpack(py_back_canny, 0, arr_kwlist[1])) return nullptr;
    if (!BuffDepth.unpack(py_depth, 0, arr_kwlist[2])) return nullptr;
    // if (!BuffOutputL.unpack(py_out_l, 0, arr_kwlist[3])) return nullptr;
    // if (!BuffOutputD.unpack(py_out_d, 0, arr_kwlist[4])) return nullptr;
    if (!BuffOutputF.unpack(py_out_f, 0, arr_kwlist[3])) return nullptr;
    if (!BuffOutputNF.unpack(py_out_Nf, 0, arr_kwlist[4])) return nullptr;
    if (!BuffOutputLImg.unpack(py_out_limg, 0, arr_kwlist[5])) return nullptr;
    if (!BuffOutputMImg.unpack(py_out_mimg, 0, arr_kwlist[6])) return nullptr;

    long int* shape_in = BuffCanny.py_buf.shape;
    // long int* shape_d = BuffDepth.py_buf.shape;
    // long int* shape_out = BuffOutputL.py_buf.shape;
    long int* shape_fout = BuffOutputF.py_buf.shape;
    long int* shape_Nfout = BuffOutputNF.py_buf.shape;
    long int* shape_limg = BuffOutputLImg.py_buf.shape;

    uint8_t *canny_host = (uint8_t*)BuffCanny.buff_arr.host;
    uint8_t *back_canny_host =  (uint8_t*)BuffBackCanny.buff_arr.host;
    uint8_t *depth_host =  (uint8_t*)BuffDepth.buff_arr.host;
    // int32_t *out_l = (int32_t*)BuffOutputL.buff_arr.host;
    // int32_t *out_d = (int32_t*)BuffOutputD.buff_arr.host;
    int32_t *out_f = (int32_t*)BuffOutputF.buff_arr.host;
    int32_t *out_Nf = (int32_t*)BuffOutputNF.buff_arr.host;
    int32_t *out_limg = (int32_t*)BuffOutputLImg.buff_arr.host;
    int32_t *out_mimg = (int32_t*)BuffOutputMImg.buff_arr.host;

    int w = (int)shape_in[1];
    int h = (int)shape_in[0];
    int block_size=16;
    Buffer<uint8_t> canny_hbuff = Buffer<uint8_t>(canny_host, (int)shape_in[1], (int)shape_in[0]);
    Buffer<uint8_t> back_canny_hbuff= Buffer<uint8_t>(back_canny_host, (int)shape_in[1], (int)shape_in[0]);
    Buffer<uint8_t> depth_hbuff= Buffer<uint8_t>(depth_host, (int)shape_in[1], (int)shape_in[0]);
    // Buffer<int32_t, 3> out_l_hbuff= Buffer<int32_t, 3>(out_l, (int)shape_out[2],(int)shape_out[1], (int)shape_out[0]);
    // Buffer<int32_t, 3> out_d_hbuff= Buffer<int32_t, 3>(out_d,(int)shape_out[2],(int)shape_out[1], (int)shape_out[0]);
    Buffer<int32_t, 3> out_f_hbuff= Buffer<int32_t, 3>(out_f,(int)shape_fout[2],(int)shape_fout[1], (int)shape_fout[0]);
    Buffer<int32_t, 1> out_Nf_hbuff= Buffer<int32_t, 1>(out_Nf,(int)shape_Nfout[0]);
    Buffer<int32_t, 2> out_limg_hbuff= Buffer<int32_t, 2>(out_limg,(int)shape_limg[1], (int)shape_limg[0]);
    Buffer<int32_t, 2> out_mimg_hbuff= Buffer<int32_t, 2>(out_mimg,(int)shape_limg[1], (int)shape_limg[0]);


    Py_BEGIN_ALLOW_THREADS
        foreground_mesh_verts(canny_hbuff, back_canny_hbuff, depth_hbuff, out_f_hbuff, out_Nf_hbuff, out_limg_hbuff, out_mimg_hbuff);
    Py_END_ALLOW_THREADS

    float result=0.0;

    return PyLong_FromLong((long)result);
}

const char* const arr_fwlist[] = {
    "I_depth",
    "P_faces",
    "I_Vlabels",
    "Vidx_init",
    "out_v_idx",
    "out_verts",
    "out_uvs",
    "out_faces",
    nullptr
    };

static PyObject* pforeground_mesh_faces(PyObject *module, PyObject *args, PyObject *kwargs){
    PyObject* py_I_depth;
    PyObject* py_P_faces;
    PyObject* py_I_Vlabels;
    int Vidx_init;
    PyObject* py_I_Vidx;
    PyObject* py_faces;
    PyObject* py_verts;
    PyObject* py_uvs;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOOiOOOO", (char**)arr_fwlist, &py_I_depth, &py_P_faces, &py_I_Vlabels, &Vidx_init, &py_I_Vidx, &py_verts, &py_uvs, &py_faces)) {
        PyErr_Format(PyExc_ValueError, "Parser Args Internal error");
        return nullptr;
    }

    PyHalideBuffer<uint8_t, 2> BuffDepth;
    PyHalideBuffer<int32_t, 3> BuffPFaces;
    PyHalideBuffer<int32_t, 2> BuffIVLabels;
    PyHalideBuffer<int32_t, 2> BuffOutputIVidx;
    PyHalideBuffer<float, 2> BuffOutputV;
    PyHalideBuffer<float, 2> BuffOutputUV;
    PyHalideBuffer<int32_t, 2> BuffOutputF;

    if (!BuffDepth.unpack(py_I_depth, 0, arr_fwlist[0])) return nullptr;
    if (!BuffPFaces.unpack(py_P_faces, 0, arr_fwlist[1])) return nullptr;
    if (!BuffIVLabels.unpack(py_I_Vlabels, 0, arr_fwlist[2])) return nullptr;
    if (!BuffOutputIVidx.unpack(py_I_Vidx, 0, arr_fwlist[4])) return nullptr;
    if (!BuffOutputV.unpack(py_verts, 0, arr_fwlist[5])) return nullptr;
    if (!BuffOutputUV.unpack(py_uvs, 0, arr_fwlist[6])) return nullptr;
    if (!BuffOutputF.unpack(py_faces, 0, arr_fwlist[7])) return nullptr;

    long int* shape_d = BuffDepth.py_buf.shape;
    long int* shape_f = BuffPFaces.py_buf.shape;
    long int* shape_vl = BuffIVLabels.py_buf.shape;
    long int* shape_vidx = BuffOutputIVidx.py_buf.shape;
    long int* shape_v = BuffOutputV.py_buf.shape;
    long int* shape_uv = BuffOutputUV.py_buf.shape;
    long int* shape_out_f = BuffOutputF.py_buf.shape;

    uint8_t *depth_host =  (uint8_t*)BuffDepth.buff_arr.host;
    int32_t *faces = (int32_t*)BuffPFaces.buff_arr.host;
    // int32_t *Nfaces = (int32_t*)BuffPNFaces.buff_arr.host;
    int32_t *vlabels = (int32_t*)BuffIVLabels.buff_arr.host;
    int32_t *out_vidx = (int32_t*)BuffOutputIVidx.buff_arr.host;
    float *out_v = (float*)BuffOutputV.buff_arr.host;
    float *out_uv = (float*)BuffOutputUV.buff_arr.host;
    int32_t *out_f = (int32_t*)BuffOutputF.buff_arr.host;

    int w = (int)shape_d[1];
    int h = (int)shape_d[0];
    int block_size=16;
    Buffer<uint8_t> depth_hbuff= Buffer<uint8_t>(depth_host, (int)shape_d[1], (int)shape_d[0]);
    Buffer<int32_t, 3> f_hbuff= Buffer<int32_t, 3>(faces, (int)shape_f[2],(int)shape_f[1], (int)shape_f[0]);
    Buffer<int32_t, 2> vlabels_hbuff= Buffer<int32_t, 2>(vlabels, (int)shape_vl[1], (int)shape_vl[0]);
    Buffer<int32_t, 2> out_vidx_hbuff= Buffer<int32_t, 2>(out_vidx,(int)shape_vidx[1], (int)shape_vidx[0]);
    Buffer<float, 2> out_v_hbuff= Buffer<float, 2>(out_v,(int)shape_v[1], (int)shape_v[0]);
    Buffer<float, 2> out_uv_hbuff= Buffer<float, 2>(out_uv,(int)shape_uv[1], (int)shape_uv[0]);
    Buffer<int32_t, 2> out_f_hbuff= Buffer<int32_t, 2>(out_f,(int)shape_out_f[1], (int)shape_out_f[0]);


    Py_BEGIN_ALLOW_THREADS
        foreground_mesh_faces(depth_hbuff, f_hbuff, vlabels_hbuff, Vidx_init, out_vidx_hbuff, out_v_hbuff, out_uv_hbuff, out_f_hbuff);
    Py_END_ALLOW_THREADS

    float result=0.0;
    result = shape_vidx[0];

    return PyLong_FromLong((long)result);
}

static PyObject* division(PyObject *self, PyObject *args) {
    long dividend, divisor;
    if (!PyArg_ParseTuple(args, "ll", &dividend, &divisor)) {
        return NULL;
    }
    if (0 == divisor) {
        PyErr_Format(PyExc_ZeroDivisionError, "Dividing %d by zero!", dividend);
        return NULL;
    }
    return PyLong_FromLong(dividend / divisor);
}



///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////


// Exported methods are collected in a table
PyMethodDef method_table[] = {
    {"pforeground_mesh_verts", (PyCFunction) pforeground_mesh_verts, METH_VARARGS | METH_KEYWORDS, nullptr},
    {"pforeground_mesh_faces", (PyCFunction) pforeground_mesh_faces, METH_VARARGS | METH_KEYWORDS, nullptr},
    {"division", (PyCFunction) division, METH_VARARGS, "Method docstring"},
    {NULL, NULL, 0, NULL} // Sentinel value ending the table
};

// A struct contains the definition of a module
PyModuleDef _module = {
    PyModuleDef_HEAD_INIT,
    "foreground_mesh_verts_module", // Module name
    "This is the module docstring",
    -1,   // Optional size of the module state memory
    method_table,
    NULL, // Optional slot definitions
    NULL, // Optional traversal function
    NULL, // Optional clear function
    NULL  // Optional module deallocation function
};

// The module init function
PyMODINIT_FUNC PyInit_foreground_mesh_verts_module(void) {
    return PyModule_Create(&_module);
}

