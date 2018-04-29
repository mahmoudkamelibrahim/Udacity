#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  //Initialise vector
  VectorXd rmse(4);
  rmse << 0,0,0,0;

  //Check if data is valid
  if(estimations.size() != ground_truth.size() ||
     estimations.size() == 0){
    std:: cout << "ERROR: Invalid estimation or ground_truth data \n";
    return rmse;
  }

  for(unsigned int i=0; i<estimations.size(); ++i){
    VectorXd residual = estimations[i] - ground_truth[i];   
    residual = residual.array()*residual.array();
    rmse += residual;    
  }

  //Compute mean
  rmse = rmse/estimations.size();

  //compute square root
  rmse = rmse.array().sqrt();

  return rmse;
}


double Tools::CalculeNIS(const VectorXd &z_diff, const MatrixXd &S_inverse){

  double epislon = 0;
  //compute error
  epislon = z_diff.transpose() * S_inverse * z_diff;

  return epislon;
}
