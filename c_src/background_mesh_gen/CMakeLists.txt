cmake_minimum_required(VERSION 3.22)
project(HalidePythonExample)

set(CMAKE_CXX_STANDARD 17)  # or newer
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(Halide REQUIRED)
set(verts_generator_src "background_verts_generator.py")
set(faces_generator_src "background_faces_generator.py")
set(output_path_vert "background_verts_gen")
set(output_path_face "background_faces_gen")

add_halide_runtime(py_run)
# add_halide_generator(background_mesh_verts.generator SOURCES ${verts_generator_src})
add_halide_generator(app_gen_background_verts SOURCES ${verts_generator_src})

# Build a Halide library as you usually would, but be sure to include `PYTHON_EXTENSION`
add_halide_library(background_mesh_verts
                   USE_RUNTIME py_run
                   FROM app_gen_background_verts
                   GENERATOR back_verts_Adams2019
                   PYTHON_EXTENSION output_path_vert
                   PLUGINS Halide::Adams2019
                   COMPILER_LOG)
# Now wrap the generated code with a Python extension.
# (Note that module name defaults to match the target name; we only
# need to specify MODULE_NAME if we need a name that may differ)

add_halide_generator(app_gen_background_faces SOURCES ${faces_generator_src})

# Build a Halide library as you usually would, but be sure to include `PYTHON_EXTENSION`
add_halide_library(background_mesh_faces
USE_RUNTIME py_run
FROM app_gen_background_faces
GENERATOR back_faces_Adams2019
PYTHON_EXTENSION output_path_face
PLUGINS Halide::Adams2019
COMPILER_LOG)
# Now wrap the generated code with a Python extension.
# (Note that module name defaults to match the target name; we only
# need to specify MODULE_NAME if we need a name that may differ)
add_halide_python_extension_library(background_verts_extension
                                    MODULE_NAME background_verts_module
                                    HALIDE_LIBRARIES background_mesh_verts background_mesh_faces)

# add_executable(background_mesh background_mesh.cpp)
# target_link_libraries(background_mesh
#                        PRIVATE
#                        background_mesh_verts
#                        background_mesh_faces
#                        Halide::ImageIO
#                        )
# target_include_directories(background_mesh PUBLIC
#                        "${PROJECT_BINARY_DIR}"
#                        )
# add_halide_python_extension_library(background_faces_extension
#                                     MODULE_NAME background_faces_module
#                                     HALIDE_LIBRARIES back_faces_generator)