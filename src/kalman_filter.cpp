#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
    x_ = x_in;
    P_ = P_in;
    F_ = F_in;
    H_ = H_in;
    R_ = R_in;
    Q_ = Q_in;
}

void KalmanFilter::Predict() {
    // x_ = F*x_ + u
    // u is 0, so x_ = F*x_
    x_ = F_ * x_;
    P_ = F_ * P_ * F_.transpose() + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
    VectorXd y = z - H_ * x_;
    UpdateHelper(y);
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
    double px = x_[0];
    double py = x_[1];
    double vx = x_[2];
    double vy = x_[3];
    double px_py_2 = sqrt(pow(x_[0],2.) + pow(x_[1],2.));

    // Calculate vector equivalent to h(x)
    VectorXd h_x = VectorXd(3);
    h_x[0] = px_py_2;
    h_x[1] = atan2(py,px);
    h_x[2] = (px*vx + py*vy)/px_py_2;

    VectorXd y = z - h_x;
    UpdateHelper(y);
}

void KalmanFilter::UpdateHelper(const Eigen::VectorXd &y) {
    MatrixXd S = H_ * P_ * H_.transpose() + R_;
    MatrixXd K = P_ * H_.transpose() * S.inverse();
    x_ = x_ + K * y;
    long x_size = x_.size();
    MatrixXd I = MatrixXd::Identity(x_size, x_size);
    P_ = (I - K * H_) * P_;
}
