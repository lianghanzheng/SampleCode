# A Simple Lowering Process

```sh
cmake -B build -S .
cmake --build build

./build/tools/sysy-opt --sysy-lower tests/matmul.mlir
```
