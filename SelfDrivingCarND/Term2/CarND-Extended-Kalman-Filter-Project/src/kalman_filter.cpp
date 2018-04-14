#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

// Please note that the Eigen library does not initialize 
// VectorXd or MatrixXd objects with zeros upon creation.

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
	x_ = F_ * x_;
	MatrixXd Ft = F_.transpose();
	P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
	VectorXd z_pred = H_ * x_;
	VectorXd y = z - z_pred;
	MatrixXd Ht = H_.transpose();
	MatrixXd S = H_ * P_ * Ht + R_;
	MatrixXd Si = S.inverse();
	MatrixXd PHt = P_ * Ht;
	MatrixXd K = PHt * Si;

	//new estimate
	x_ = x_ + (K * y);
	long x_size = x_.size();
	MatrixXd I = MatrixXd::Identity(x_size, x_size);
	P_ = (I - K * H_) * P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {

  //get state vector
  double px = x_[0];
  double py = x_[1];
  double vx = x_[2];
  double vy = x_[3];

  /**
   *transform state vector to polar coordinates  
   *check for division by zero
   */
  if(fabs(px) < 0.0001){
    px = 0.0001;
  }
  double rho_pred    =  sqrt(px*px + py*py);
  double phi_pred = atan2(py,px);    
  if(fabs(rho_pred) < 0.0001){
    rho_pred = 0.0001;
  }   
  double rho_dt_pred = (px*vx + py*vy)/rho_pred;
    
  VectorXd z_pred(3);
  z_pred << rho_pred, phi_pred, rho_dt_pred;
  
  VectorXd y = z - z_pred;
  
  //normalize phi between -pi and pi
  double phi = y[1];
  phi = KalmanFilter::Normalize(phi, PI, -PI);
  y(1) = phi;

  //get kalman filter gain K
	MatrixXd Ht = H_.transpose();
	MatrixXd S = H_ * P_ * Ht + R_;
	MatrixXd Si = S.inverse();
	MatrixXd PHt = P_ * Ht;
	MatrixXd K = PHt * Si;

	//new estimate
	x_ = x_ + (K * y);
	long x_size = x_.size();
	MatrixXd I = MatrixXd::Identity(x_size, x_size);
	P_ = (I - K * H_) * P_;
}

double KalmanFilter::Normalize(double value, double max, double min){
  value -= min;  
  max -= min;
  if (max == 0)
    return min;
  value = fmod(value, max);
  value += min;
  while (value < min){
    value += max;
  }
  return value;
}
