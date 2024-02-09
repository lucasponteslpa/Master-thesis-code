rm -r build/
cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="/home/lpa1/Halide-install" -S . -B build
cmake --build ./build
mv build/depth_enhance_module.cpython-38-x86_64-linux-gnu.so ./
