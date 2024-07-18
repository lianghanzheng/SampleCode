func.func @test_matmul_f32(%C: memref<2048x2048xf32>, %A: memref<2048x2048xf32>, %B: memref<2048x2048xf32>) {
  sysy.matmul %C, %A, %B : memref<2048x2048xf32>, memref<2048x2048xf32>, memref<2048x2048xf32>
  func.return
}


func.func @test_matmul_int(%C: memref<2048x2048xi32>, %A: memref<2048x2048xi32>, %B: memref<2048x2048xi32>) {
  sysy.matmul %C, %A, %B : memref<2048x2048xi32>, memref<2048x2048xi32>, memref<2048x2048xi32>
  func.return
}
