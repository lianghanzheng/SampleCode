# A Simple Lowering Process

```sh
cmake -B build -S .
cmake --build build

./build/tools/sysy-opt --sysy-lower tests/matmul.mlir
```

针对新加入的从tensor的降级的命令如下：

```sh
./build/tools/sysy-opt --sysy-tensor-to-memref --sysy-lower tests/matmul-tensor.mlir
```