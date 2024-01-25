cmake_minimum_required(VERSION 3.22)
project(merge_generator)

# Set up language settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)

# Find Halide
find_package(Halide REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)

add_halide_generator(merge_mesh_verts.generator SOURCES MergeMeshVerts.cpp)

add_halide_library(merge_mesh_verts FROM merge_mesh_verts.generator
                   C_BACKEND
                   GENERATOR merge_mesh_verts
                   PYTHON_EXTENSION output_path_var)

add_halide_library(merge_mesh_verts_native FROM merge_mesh_verts.generator
                   GENERATOR merge_mesh_verts)

add_halide_generator(merge_mesh_faces.generator SOURCES MergeMeshFaces.cpp)

add_halide_library(merge_mesh_faces FROM merge_mesh_faces.generator
                    C_BACKEND
                    GENERATOR merge_mesh_faces
                    PYTHON_EXTENSION output_path_var)

add_halide_library(merge_mesh_faces_native FROM merge_mesh_faces.generator
                    GENERATOR merge_mesh_faces)


set(PY_SRCS merge_pymodule.cpp merge_ext.cpp)
# set(PY_SRCS merge_pymodule.cpp)
Python3_add_library(merge_mesh_verts_module MODULE WITH_SOABI ${PY_SRCS})
target_link_libraries(merge_mesh_verts_module PRIVATE
                      merge_mesh_verts
                      merge_mesh_verts_native
                      merge_mesh_faces
                      merge_mesh_faces_native
                      Halide::Halide
                      Halide::Runtime
                      Halide::Tools
                      )
set_target_properties(
    merge_mesh_verts_module
    PROPERTIES
        PREFIX ""
        OUTPUT_NAME "merge_mesh_verts_module"
        LINKER_LANGUAGE C
    )
target_include_directories(merge_mesh_verts_module PUBLIC
    "${PROJECT_BINARY_DIR}"
    )



# add_executable(run_c run.cpp quadtree_ext.cpp)
# target_link_libraries(run_c
#                        PRIVATE
#                        merge_mesh_verts
#                        merge_mesh_verts_native
#                        Halide::ImageIO
#                        )
# target_include_directories(run_c PUBLIC
#                            "${PROJECT_SOURCE_DIR}"
#                            )
