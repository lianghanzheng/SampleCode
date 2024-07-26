func.func @test_matmul_f32() {
  %A = "sysy.constant"() {value = dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf32>} : () -> tensor<2x3xf32>
  %B = "sysy.constant"() {value = dense<[[6.0, 5.0], [4.0, 3.0], [2.0, 1.0]]> : tensor<3x2xf32>} : () -> tensor<3x2xf32>
  %C = "sysy.constant"() {value = dense<[[0.0, 0.0], [0.0, 0.0]]> : tensor<2x2xf32>} : () -> tensor<2x2xf32>

  sysy.matmul %C, %A, %B : tensor<2x2xf32>, tensor<2x3xf32>, tensor<3x2xf32>
  func.return
}

//func.func @test_matmul_i32() {
//  %A = arith.constant dense<[[1, 2, 3], [4, 5, 6]]> : tensor<2x3xi32>
//  %B = arith.constant dense<[[6, 5], [4, 3], [2, 1]]> : tensor<3x2xi32>
//  %C = arith.constant dense<[[0, 0], [0, 0]]> : tensor<2x2xi32>
//
//  sysy.matmul %C, %A, %B : tensor<2x2xi32>, tensor<2x3xi32>, tensor<3x2xi32>
//  func.return
//}