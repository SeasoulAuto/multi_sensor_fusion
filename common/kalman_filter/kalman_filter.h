 
 #include <Eigen/Dense>
 
 namespace MultiSensorFusion {
 namespace Common {


 // 一个基础通用的卡尔曼滤波器   
 class KalmanFilter {
    public:
        KalmanFilter(int state_dim, int measurement_dim, int control_dim) : 
            A_(state_dim, state_dim),
            B_(state_dim, control_dim),
            covariance_(state_dim, state_dim),
            H_(measurement_dim, state_dim),
            measurement_noise_(measurement_dim, measurement_dim),
            x_(state_dim),
            state_dim_(state_dim),
            measurement_dim_(measurement_dim),
            control_dim_(control_dim)
            {}
        ~KalmanFilter() = default;
    private:
        // 存储初始化的状态矩阵x0, 以及初始化的协方差矩阵p0
        void Init(const Eigen::VectorXd & x0);
       // 预测阶段， X(k+1) = A * X(k) + B * U(k) 
       void Predict(const Eigen::VectorXd & u);
       void Update(const Eigen::VectorXd & z);

       // 预测阶段

       // 状态转移矩阵(维度： state_dim * state_dim)
       Eigen::MatrixXd A_;
       // 控制输入矩阵(维度: state_dim * control_dim)
       Eigen::MatrixXd B_;
       // 当前的状态矩阵(维度: state_dim * 1), Eigen中 VectorXd是列向量
       Eigen::VectorXd x_;

       // 协方差矩阵(维度: state_dim * state_dim)
       Eigen::MatrixXd covariance_;
       // 过程噪声协方差矩阵  Q(维度： state_dim * state_dim)
       Eigen::MatrixXd process_noise_;

       // 更新阶段
       // 观测矩阵(维度： measurement_dim * state_dim) 
       Eigen::MatrixXd H_; 
       // 卡尔曼增益(维度：state_dim * measurement_dim)
       Eigen::MatrixXd kalman_gain_;
       // 观测噪声协方差矩阵(维度： measurement_dim * measurement_dim)  R
       Eigen::MatrixXd measurement_noise_;

       // 存储 state_dim, measurement_dim, control_dim
       int state_dim_;
       int measurement_dim_;
       int control_dim_;

 }; // class KalmanFilter
 } // namespace Common
 } // namespace MultiSensorFusion
