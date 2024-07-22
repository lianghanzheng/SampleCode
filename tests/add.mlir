func.func @test_add_f32_2d(%C: memref<2048x2048xf32>, %A: memref<2048x2048xf32>, %B: memref<2048x2048xf32>) {
  sysy.add %C, %A, %B : memref<2048x2048xf32>, memref<2048x2048xf32>, memref<2048x2048xf32>
  func.return
}

//-----------------

func.func @test_add_f32_1d(%C: memref<2048xf32>, %A: memref<2048xf32>, %B: memref<2048xf32>) {
  sysy.add %C, %A, %B : memref<2048xf32>, memref<2048xf32>, memref<2048xf32>
  func.return
}

//-----------------

func.func @test_add_f32_3d(%C: memref<2048x2048x2048xf32>, %A: memref<2048x2048x2048xf32>, %B: memref<2048x2048x2048xf32>) {
  sysy.add %C, %A, %B : memref<2048x2048x2048xf32>, memref<2048x2048x2048xf32>, memref<2048x2048x2048xf32>
  func.return
}

//-----------------

func.func @test_add_int_2d(%C: memref<2048x2048xi32>, %A: memref<2048x2048xi32>, %B: memref<2048x2048xi32>) {
  sysy.add %C, %A, %B : memref<2048x2048xi32>, memref<2048x2048xi32>, memref<2048x2048xi32>
  func.return
}