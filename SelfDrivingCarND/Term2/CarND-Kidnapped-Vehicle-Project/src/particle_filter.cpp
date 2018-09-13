/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	num_particles = 100;

  // resize weights and particles vector
  weights.resize(num_particles);
  particles.resize(num_particles);

  // define random number generation engine
  random_device rd;
  default_random_engine gen(rd());

  // define noise
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

  // create particles
  for(int i = 0; i < num_particles; ++i) {
    particles[i].id = i;
    particles[i].x = dist_x(gen);
    particles[i].y = dist_y(gen);
    particles[i].theta = dist_theta(gen);
    particles[i].weight = 1.0;
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {

  default_random_engine gen;

  // define noise
  normal_distribution<double> dist_x(0, std_pos[0]);
  normal_distribution<double> dist_y(0, std_pos[1]);
  normal_distribution<double> dist_theta(0, std_pos[2]);

  for (int i = 0; i < num_particles; ++i) {
    if (fabs(yaw_rate) < 0.00001) {
      particles[i].x += velocity * delta_t * cos(particles[i].theta);
      particles[i].y += velocity * delta_t * sin(particles[i].theta);
    }
    else {
        double yR_D =yaw_rate * delta_t;
        double v_YR=velocity/yaw_rate;
        double sin_term1=sin(particles[i].theta+yR_D) ;
        double sin_term2=sin(particles[i].theta);
        double cos_term1=cos(particles[i].theta);
        double cos_term2=cos(particles[i].theta+ yR_D);
        particles[i].x+= v_YR*(sin_term1-sin_term2) ;
        particles[i].y+= v_YR*(cos_term1-cos_term2) ;
        particles[i].theta+= yR_D;
}

            particles[i].x += dist_x(gen);
            particles[i].y += dist_y(gen);
            particles[i].theta += dist_theta(gen);
  }}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
  // start looping over observations
  for(int i = 0; i < observations.size(); ++i) {
    LandmarkObs obs = observations[i];
    double min_dist = numeric_limits<double>::max();
    int map_id = -1;

    // start looping over predicted measurements
    for(int j = 0; j < predicted.size(); ++j) {
      LandmarkObs pred = predicted[j];
      double cur_dist = dist(obs.x, obs.y, pred.x, pred.y);

      // get the current closest particle's id
      if (cur_dist < min_dist) {
        min_dist = cur_dist;
        map_id = j;
      }
    }

    // assign closest particle id to the observation
    observations[i].id = map_id;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	

       double sig_x_2,sig_y_2;
        sig_x_2= std_landmark[0]*std_landmark[0];
        sig_y_2= std_landmark[1]*std_landmark[1];

        // calculate normalization term
        double gauss_norm= 1/( 2 * M_PI * std_landmark[0] * std_landmark[1]); //  s_x,s_y #TESTMA7MOUDK


// for each particle...
  for (unsigned int i = 0; i < num_particles; i++) {

    // get the particle x, y coordinates
    double p_x = particles[i].x;
    double p_y = particles[i].y;
    double p_theta = particles[i].theta;

    // create a vector to hold the map landmark locations predicted to be within sensor range of the particle
    vector<LandmarkObs> predictions;

    // for each map landmark...
    for (unsigned int j = 0; j < map_landmarks.landmark_list.size(); j++) {

      // get id and x,y coordinates
      float lm_x = map_landmarks.landmark_list[j].x_f;
      float lm_y = map_landmarks.landmark_list[j].y_f;
      int lm_id = map_landmarks.landmark_list[j].id_i;
      
      // only consider landmarks within sensor range of the particle (rather than using the "dist" method considering a circular 
      // region around the particle, this considers a rectangular region but is computationally faster)
      if(dist(lm_x, lm_y, p_x, p_y) <= sensor_range) {

        // add prediction to vector
        predictions.push_back(LandmarkObs{ lm_id, lm_x, lm_y });
      }
    }

    // create and populate a copy of the list of observations transformed from vehicle coordinates to map coordinates

    if(predictions.size() == 0) {
      particles[i].weight = 0;
      weights[i] = 0;
    }
    else {

    vector<LandmarkObs> transformed_os;
    for (unsigned int j = 0; j < observations.size(); ++j) {
      double t_x = cos(p_theta)*observations[j].x - sin(p_theta)*observations[j].y + p_x;
      double t_y = sin(p_theta)*observations[j].x + cos(p_theta)*observations[j].y + p_y;
      transformed_os.push_back(LandmarkObs{ observations[j].id, t_x, t_y });
    }

    // perform dataAssociation for the predictions and transformed observations on current particle

    dataAssociation(predictions, transformed_os);
////////////%%//////////
      double total_prob = 1.0;
      for(int l = 0; l < transformed_os.size(); ++l) {

        auto obs = transformed_os[l];
        auto predicted = predictions[obs.id];

        double dx = obs.x - predicted.x;
        double dy = obs.y - predicted.y;

        double exponent= (dx * dx / (2 * sig_x_2) + dy * dy / (2 * sig_y_2));

        total_prob *= gauss_norm * exp(-exponent);
      }
      particles[i].weight = total_prob;
      weights[i] = total_prob;
    }
  }
}


void ParticleFilter::resample() {

  
  // Vector for new particles
  vector<Particle> new_particles (num_particles);
  
  // Use discrete distribution to return particles by weight
  random_device rd;
  default_random_engine gen(rd());
  for (int i = 0; i < num_particles; ++i) {
    discrete_distribution<int> index(weights.begin(), weights.end());
    new_particles[i] = particles[index(gen)];
    
  }
  
  // Replace old particles with the resampled particles
  particles = new_particles;}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
s = s.substr(0, s.length()-1); // get rid of the trailing space
    return s;
}
