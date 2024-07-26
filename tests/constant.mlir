func.func @test_f32() {
  %1 = "sysy.constant"() {value = dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf32>} : () -> tensor<2x3xf32>
  func.return
}

//func.func @test_i32() {
////  %0 = sysy.constant [[1, 2, 3], [4, 5, 6]]> : tensor<2x3xi32>
//  %1 = "sysy.constant"() {value = dense<[[1, 2, 3], [4, 5, 6]]> : tensor<2x3xi32>} : () -> tensor<2x3xi32>
//  func.return
//}