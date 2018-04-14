#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;
  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  Hj_ = MatrixXd(3, 4);
  MatrixXd F_ = MatrixXd(4,4);
  MatrixXd P_ = MatrixXd(4,4);
  VectorXd x_ = VectorXd(4);
  MatrixXd Q_ = MatrixXd(4,4);
  MatrixXd R_ = MatrixXd(2, 2);
  MatrixXd H_ = MatrixXd(2, 4); 

  //measurement covariance matrix - laser
  R_laser_ << 0.0225, 0,
        0, 0.0225;

  //measurement covariance matrix - radar
  R_radar_ << 0.06, 0, 0,
        0, 0.0006, 0,
        0, 0, 0.06;

  //initialize state vector
  x_ << 1.0,1.0,1.0,1.0;

  //Laser measurement matrix
  H_laser_ << 1,0,0,0,
    0,1,0,0;

  //radar jacobian matrix for Extended Filter
  Hj_ << 0,0,0,0,
    0,0,0,0,
    0,0,0,0;

  //state transition matrix
  F_ << 1,0,1,0,
    0,1,0,1,
    0,0,1,0,
    0,0,0,1;
  
  //measurement noise
  P_ << 1,0,0,0,
    0,1,0,0,
    0,0,1000,0,
    0,0,0,1000;

  //process noise
  Q_ << 0,0,0,0,
    0,0,0,0,
    0,0,0,0,
    0,0,0,0;
  
  //Init Kalman Filter
  ekf_.Init(x_, P_, F_, H_laser_, R_laser_, Q_);
}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {


  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {
    /**
    TODO:
      * Initialize the state ekf_.x_ with the first measurement.
      * Create the covariance matrix.
      * Remember: you'll need to convert radar from polar to cartesian coordinates.
    */
    // first measurement
    cout << "EKF: " << endl;
  //  ekf_.x_ = VectorXd(4);
  //  ekf_.x_ << 1, 1, 1, 1;

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      cout << "RADAR measurements\n";
      /**
       * Convert radar from polar to cartesian coordinates and initialize state.
       * speed can be ignored for the first read 
      */
      double rho    = measurement_pack.raw_measurements_(0);
      double phi = measurement_pack.raw_measurements_(1);
      
      //convert to cartesian coordinates
      double px = rho * cos(phi);
      double py = rho * sin(phi);      
      ekf_.x_ << px, py, 0.0, 0.0;
    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      cout << "LASER measurements\n";
      
      double px = measurement_pack.raw_measurements_(0);
      double py = measurement_pack.raw_measurements_(1);
      ekf_.x_ << px, py, 0.0, 0.0;
	
    }

    // done initializing, no need to predict or update
    is_initialized_ = true;

    //save previous timestamp
    previous_timestamp_ = measurement_pack.timestamp_;

    return;
  }

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/

  /**
     * Update the state transition matrix F according to the new elapsed time.
      - Time is measured in seconds.
     * Update the process noise covariance matrix.
     * Use noise_ax = 9 and noise_ay = 9 for your Q matrix.
   */

  double dt = (measurement_pack.timestamp_ - previous_timestamp_)/ 1000000.0L;
  previous_timestamp_ = measurement_pack.timestamp_;

  //update state transition F
  ekf_.F_ << 1, 0, dt, 0,
    0, 1, 0, dt,
    0, 0, 1, 0,
    0, 0, 0, 1;

  //helper values to compute Q_
  double dt_2 = dt * dt;
  double dt_3 = dt_2 * dt;
  double dt_4 = dt_3 * dt;

  //predefined noise values
  double noise_ax = 9.0;
  double noise_ay = 9.0;

  //update process noise covariance matrix Q  
  ekf_.Q_ << dt_4/4*noise_ax, 0, dt_3/2*noise_ax, 0,
    0, dt_4/4*noise_ay, 0, dt_3/2*noise_ay,
    dt_3/2*noise_ax, 0, dt_2*noise_ax, 0,
    0, dt_3/2*noise_ay, 0, dt_2*noise_ay;

  ekf_.Predict();

  /*****************************************************************************
   *  Update
   ****************************************************************************/

  /**
     * Use the sensor type to perform the update step.
     * Update the state and covariance matrices.
   */

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // Radar updates
    double rho = measurement_pack.raw_measurements_[0];
    double phi = measurement_pack.raw_measurements_[1];
    double rho_dt = measurement_pack.raw_measurements_[2];

    VectorXd input(3);
    input << rho, phi, rho_dt;
    
    Tools t;    
    ekf_.H_ = t.CalculateJacobian(ekf_.x_);
    ekf_.R_ = R_radar_;
    ekf_.UpdateEKF(input);
  } else {
    // Laser updates
    double px = measurement_pack.raw_measurements_[0];
    double py = measurement_pack.raw_measurements_[1];

    VectorXd input(2);
    input << px, py;
    
    ekf_.H_ = H_laser_;
    ekf_.R_ = R_laser_;
    ekf_.Update(input);
  }

  // print the output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}
