#[burn_tensor_testgen::testgen(ad_reshape)]
mod tests {
    use super::*;
    use burn_tensor::Data;

    #[::tracing_test::traced_test]
#[test]
    fn should_diff_reshape() {
        let data_1: Data<f32, 2> = Data::from([[1.0, 7.0], [2.0, 3.0]]);
        let data_2: Data<f32, 1> = Data::from([4.0, 7.0, 2.0, 3.0]);

        let device = Default::default();
        let tensor_1 = TestAutodiffTensor::from_data(data_1, &device).require_grad();
        let tensor_2 = TestAutodiffTensor::from_data(data_2, &device).require_grad();

        let tensor_3 = tensor_2.clone().reshape([2, 2]);
        let tensor_4 = tensor_1.clone().matmul(tensor_3);
        let grads = tensor_4.backward();

        let grad_1 = tensor_1.grad(&grads).unwrap();
        let grad_2 = tensor_2.grad(&grads).unwrap();

        assert_eq!(grad_1.to_data(), Data::from([[11.0, 5.0], [11.0, 5.0]]));
        assert_eq!(grad_2.to_data(), Data::from([3.0, 3.0, 10.0, 10.0]));
    }
}
