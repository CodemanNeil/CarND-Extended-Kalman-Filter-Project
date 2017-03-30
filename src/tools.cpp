#include <iostream>
#include "tools.h"
#include "DivideByZeroJacobianError.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
    VectorXd rmse(4);
    rmse << 0,0,0,0;

    // check the validity of the following inputs:
    //  * the estimation vector size should not be zero
    //  * the estimation vector size should equal ground truth vector size
    if (estimations.size() != ground_truth.size() || estimations.size() == 0) {
        throw std::runtime_error("Ground Truth size doesn't match estimation size");
    }

    //accumulate squared residuals
    for(unsigned int i=0; i < estimations.size(); ++i){

        VectorXd residual = estimations[i] - ground_truth[i];

        //coefficient-wise multiplication
        residual = residual.array()*residual.array();
        rmse += residual;
    }

    //calculate the mean
    rmse = rmse/estimations.size();

    //calculate the squared root
    rmse = rmse.array().sqrt();

    //return the result
    return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd &x_state) {
    MatrixXd Hj(3, 4);
    //recover state parameters
    double px = x_state(0);
    double py = x_state(1);
    double vx = x_state(2);
    double vy = x_state(3);

    //TODO: YOUR CODE HERE

    double px_2 = pow(px, 2);
    double py_2 = pow(py, 2);
    double px_py_2 = px_2 + py_2;

    //check division by zero
    if (std::fabs(px_py_2) < 0.0001) {
        throw DivideByZeroJacobianError();
    } else {
        //compute the Jacobian matrix
        Hj(0, 0) = px / sqrt(px_py_2);
        Hj(0, 1) = py / sqrt(px_py_2);
        Hj(0, 2) = 0.;
        Hj(0, 3) = 0.;
        Hj(1, 0) = -py / px_py_2;
        Hj(1, 1) = px / px_py_2;
        Hj(1, 2) = 0.;
        Hj(1, 3) = 0.;
        Hj(2, 0) = py * (vx * py - vy * px) / pow(px_py_2, 3. / 2.);
        Hj(2, 1) = px * (vy * px - vx * py) / pow(px_py_2, 3. / 2.);
        Hj(2, 2) = px / sqrt(px_py_2);
        Hj(2, 3) = py / sqrt(px_py_2);
    }
    return Hj;
}
