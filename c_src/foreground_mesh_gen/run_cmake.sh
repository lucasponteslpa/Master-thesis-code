rm -r build/
cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="/home/lpa1/Halide-install" -S . -B build
cmake --build ./build
# ./build/run_c canny.png back_canny.png depth_quant.png
cp build/foreground_mesh_verts_module.cpython-38-x86_64-linux-gnu.so ./
python test_foreground_verts_gen.py