#include "kalman_filter.h"

namespace MultiSensorFusion {
namespace Common {


// todo(wxt): Q H R z 这些都是怎么用的


// 初始化状态量
void KalmanFilter::Init(const Eigen::VectorXd & x0) {
    x_ = x0;
    process_noise_ = Eigen::MatrixXd::Identity(state_dim_, state_dim_);
    measurement_noise_ = Eigen::MatrixXd::Identity(measurement_dim_, measurement_dim_);   
}

void KalmanFilter::Predict(const Eigen::VectorXd& u) {
  x_ = A_ * x_ + B_ * u;
  covariance_ = A_ * covariance_ * A_.transpose() + process_noise_;
}

// z: 时间k的观测值
// todo(wxt): 这个z要怎么理解？
void KalmanFilter::Update(const Eigen::VectorXd& z) {
  // 1、计算卡尔曼增益
  kalman_gain_ = covariance_ * H_.transpose() *
                 (H_ * covariance_ * H_.transpose() + measurement_noise_).inverse();
  // 2、更新状态估计值
  x_ = x_ + kalman_gain_ * (z - H_ * x_);
  // 3、更新协方差矩阵
  covariance_ =
      (Eigen::MatrixXd::Identity(state_dim_, state_dim_) - kalman_gain_ * H_) *
      covariance_;
}

}  // namespace Common
}  // namespace MultiSensorFusion