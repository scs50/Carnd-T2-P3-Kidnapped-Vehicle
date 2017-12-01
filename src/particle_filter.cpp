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
using vector_t = std::vector<double>;

// Need a random engine for noise
static default_random_engine gen;

//small number
const float eps = 0.00001;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	num_particles = 201;

	//Add Normal Dist for Sensor Noise for each partice
	normal_distribution<double> N_x_init(0, std[0]);
	normal_distribution<double> N_y_init(0, std[1]);
	normal_distribution<double> N_theta_init(0, std[2]);

	//Initialize particles
	for (int i = 0; i < num_particles; i++){
		Particle p;
		p.id = i;
		p.x = x;
		p.y = y;
		p.theta = theta;
		p.weight = 1.0;

		//Add noise from norm dist above
		p.x += N_x_init(gen);
		p.y += N_y_init(gen);
		p.theta += N_theta_init(gen);

		// add to end to vector
		particles.push_back(p);
	}

	is_initialized = true;

}

const vector_t CTRV_Model(double delta_t, const vector_t& x, const vector_t& u, const double std_pos[])
{
  static default_random_engine gen;
  normal_distribution<double> dist_x(0, std_pos[0]);
  normal_distribution<double> dist_y(0, std_pos[1]);
  normal_distribution<double> dist_yaw(0, std_pos[2]);

  //extract values for better readability
  double p_x = x[0];
  double p_y = x[1];
  double yaw = x[2];

  double v = u[0];
  double yawd = u[1];

  //predicted state values
  double px_p, py_p;

  //avoid division by zero
  if (std::fabs(yawd) > 0.01) {
    px_p = p_x + v/yawd * ( sin (yaw + yawd*delta_t) - sin(yaw));
    py_p = p_y + v/yawd * ( cos(yaw) - cos(yaw+yawd*delta_t) );
  }
  else {
    px_p = p_x + v*delta_t*cos(yaw);
    py_p = p_y + v*delta_t*sin(yaw);
  }
  // Yaw angle
  double pyaw_p = yaw + yawd*delta_t;

  // Add noise
  px_p = px_p + dist_x(gen);
  py_p = py_p + dist_y(gen);
  pyaw_p = pyaw_p + dist_yaw(gen);

  //return predicted state
  return {px_p, py_p, pyaw_p};
}


void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	for (int i = 0; i < num_particles; i++) {
	//Call CTRV_model for each particle
	vector_t x_pred = CTRV_Model(delta_t, {particles[i].x, particles[i].y, particles[i].theta}, {velocity, yaw_rate}, std_pos);
    particles[i].x = x_pred[0];
    particles[i].y = x_pred[1];
    particles[i].theta = x_pred[2];

	}

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

	//Start with largest min dist
	double min_dist, dist, dx, dy;
	//ID placeholder
	int min_i;

	for (unsigned int i = 0; i < observations.size(); i++){

		//Start with current observation
		LandmarkObs o = observations[i];

		min_dist = numeric_limits<double>::max();
		min_i = -1;


		for ( unsigned int j=0; j < predicted.size(); j++){
			//Start with current prediction
			LandmarkObs p = predicted[j];

			//Calc dist between current and predicted landmarks
			dx = (p.x - o.x);
		    dy = (p.y - o.y);
		    dist = dx*dx + dy*dy;

			//Find predicted landmark nearest to observed landmark
			if (dist < min_dist) {
				min_dist = dist;
				min_i = p.id;

			}

		}

		//update observed id to nearest pred landmark
		observations[i].id = min_i;
	}

}


void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	// for each particle...
	  for (int i = 0; i < num_particles; i++) {

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
	      
	      // only consider landmarks within sensor range of the particle
	      if (fabs(lm_x - p_x) <= sensor_range && fabs(lm_y - p_y) <= sensor_range) {

	        // add prediction to vector
	        predictions.push_back(LandmarkObs{ lm_id, lm_x, lm_y });
	      }
	    }

	    // create list of observations transformed from vehicle coordinates to map coordinates
	    vector<LandmarkObs> transformed_os;
	    for (unsigned int j = 0; j < observations.size(); j++) {
	      double t_x = cos(p_theta)*observations[j].x - sin(p_theta)*observations[j].y + p_x;
	      double t_y = sin(p_theta)*observations[j].x + cos(p_theta)*observations[j].y + p_y;

	      transformed_os.push_back(LandmarkObs{ observations[j].id, t_x, t_y });
	    }

	    // dataAssociation for the predictions and transformed observations on current particle
	    dataAssociation(predictions, transformed_os);

	    // reinit weight
	    particles[i].weight = 1.0;

	    for (unsigned int j = 0; j < transformed_os.size(); j++) {
	      
	      // placeholders for observation and associated prediction coordinates
	      double o_x, o_y, pr_x, pr_y;
	      o_x = transformed_os[j].x;
	      o_y = transformed_os[j].y;

	      int associated_prediction = transformed_os[j].id;

	      // get the x,y coordinates of the prediction associated with the current observation
	      for (unsigned int k = 0; k < predictions.size(); k++) {
	        if (predictions[k].id == associated_prediction) {
	          pr_x = predictions[k].x;
	          pr_y = predictions[k].y;
	        }
	      }

	      // calculate weight for this observation with multivariate Gaussian
	      double s_x = std_landmark[0];
	      double s_y = std_landmark[1];
		  double cov_x = s_x*s_x;
		  double cov_y = s_y*s_y;
		  double normalizer = 2.0*M_PI*s_x*s_y;
		  double dx = (o_x- pr_x);
		  double dy = (o_y - pr_y);
		  double obs_w = exp(-(dx*dx/(2*cov_x) + dy*dy/(2*cov_y)))/normalizer;

	      // product of this obersvation weight with total observations weight
	      particles[i].weight *= obs_w;
	    }
	  }
	}


void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	vector<Particle> new_particles;

	// get all of the current weights
	vector<double> weights;
	for (int i = 0; i < num_particles; i++) {
	weights.push_back(particles[i].weight);
	}

	// generate random starting index for resampling wheel
	uniform_int_distribution<int> uniintdist(0, num_particles-1);
	auto index = uniintdist(gen);

	// get max weight
	double max_weight = *max_element(weights.begin(), weights.end());

	// uniform random distribution [0.0, max_weight)
	uniform_real_distribution<double> unirealdist(0.0, max_weight);

	double beta = 0.0;

	// spin the resample wheel!
	for (int i = 0; i < num_particles; i++) {
	beta += unirealdist(gen) * 2.0;
	while (beta > weights[index]) {
	  beta -= weights[index];
	  index = (index + 1) % num_particles;
	}
	new_particles.push_back(particles[index]);
	}

	particles = new_particles;
}


Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    //Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;

    return particle;
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
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
