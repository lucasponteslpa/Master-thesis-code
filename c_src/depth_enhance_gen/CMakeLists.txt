cmake_minimum_required(VERSION 3.22)
project(HalideDepthEnhance)

set(CMAKE_CXX_STANDARD 17)  # or newer
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(Halide REQUIRED)
set(depth_enhance_generator_src "DepthEnhance.cpp")

# add_halide_generator(background_mesh_verts.generator SOURCES ${verts_generator_src})
add_halide_runtime(py_run)
add_halide_generator(depth_enhance.generator SOURCES DepthEnhance.cpp)

# Build a Halide library as you usually would, but be sure to include `PYTHON_EXTENSION`
add_halide_library(depth_enhance_gen
                   USE_RUNTIME py_run
                   FROM depth_enhance.generator
                   GENERATOR depth_enhance
                   PYTHON_EXTENSION output_path_vert
                   PLUGINS Halide::Adams2019
                   AUTOSCHEDULER Halide::Adams2019
                   COMPILER_LOG)
# Now wrap the generated code with a Python extension.
# (Note that module name defaults to match the target name; we only
# need to specify MODULE_NAME if we need a name that may differ)

# Now wrap the generated code with a Python extension.
# (Note that module name defaults to match the target name; we only
# need to specify MODULE_NAME if we need a name that may differ)
add_halide_python_extension_library(depth_enhance_extension
                                    MODULE_NAME depth_enhance_module
                                    HALIDE_LIBRARIES depth_enhance_gen)

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