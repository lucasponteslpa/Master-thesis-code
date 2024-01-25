cmake_minimum_required(VERSION 3.22)
project(c_backend)

# Set up language settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)

# Find Halide
find_package(Halide REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)

add_halide_generator(foreground_mesh_verts.generator SOURCES ForegroundMeshVerts.cpp)

add_halide_library(foreground_mesh_verts FROM foreground_mesh_verts.generator
                   C_BACKEND
                   GENERATOR foreground_mesh_verts
                   PYTHON_EXTENSION output_path_var)

add_halide_library(foreground_mesh_verts_native FROM foreground_mesh_verts.generator
                   GENERATOR foreground_mesh_verts)

add_halide_generator(foreground_mesh_faces.generator SOURCES ForegroundMeshFaces.cpp)

add_halide_library(foreground_mesh_faces FROM foreground_mesh_faces.generator
                    C_BACKEND
                    GENERATOR foreground_mesh_faces
                    PYTHON_EXTENSION output_path_var)

add_halide_library(foreground_mesh_faces_native FROM foreground_mesh_faces.generator
                    GENERATOR foreground_mesh_faces)


set(PY_SRCS foreground_pymodule.cpp quadtree_ext.cpp)
Python3_add_library(foreground_mesh_verts_module MODULE WITH_SOABI ${PY_SRCS})
target_link_libraries(foreground_mesh_verts_module PRIVATE
                      foreground_mesh_verts
                      foreground_mesh_verts_native
                      foreground_mesh_faces
                      foreground_mesh_faces_native
                      Halide::Halide
                      Halide::Runtime
                      Halide::Tools
                      )
set_target_properties(
    foreground_mesh_verts_module
    PROPERTIES
        PREFIX ""
        OUTPUT_NAME "foreground_mesh_verts_module"
        LINKER_LANGUAGE C
    )
target_include_directories(foreground_mesh_verts_module PUBLIC
    "${PROJECT_BINARY_DIR}"
    )



# add_executable(run_c run.cpp quadtree_ext.cpp)
# target_link_libraries(run_c
#                        PRIVATE
#                        foreground_mesh_verts
#                        foreground_mesh_verts_native
#                        Halide::ImageIO
#                        )
# target_include_directories(run_c PUBLIC
#                            "${PROJECT_SOURCE_DIR}"
#                            )
