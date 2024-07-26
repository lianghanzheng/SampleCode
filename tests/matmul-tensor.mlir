func.func @test_matmul_f32(%C: tensor<2048x2048xf32>, %A: tensor<2048x2048xf32>, %B: tensor<2048x2048xf32>) {
  sysy.matmul %C, %A, %B : tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>
  func.return
}


func.func @test_matmul_int(%C: tensor<2048x2048xi32>, %A: tensor<2048x2048xi32>, %B: tensor<2048x2048xi32>) {
  sysy.matmul %C, %A, %B : tensor<2048x2048xi32>, tensor<2048x2048xi32>, tensor<2048x2048xi32>
  func.return
}