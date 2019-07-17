#ifndef matching2D_hpp
#define matching2D_hpp

#include <stdio.h>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>

#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "dataStructures.h"


void detKeypointsHarris(float &elapsed_time ,std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis=false);
void detKeypointsShiTomasi(float &elapsed_time , std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis=false);

// add 5 function headers //->  FAST, BRISK, ORB, AKAZE, SIFT
void detKeypointsFAST(float &elapsed_time ,std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis=false);
void detKeypointsBRISK(float &elapsed_time ,std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis=false);
void detKeypointsORB(float &elapsed_time ,std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis=false);
void detKeypointsAKAZE(float &elapsed_time ,std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis=false);
void detKeypointsSIFT(float &elapsed_time ,std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis=false);

void detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, std::string detectorType, bool bVis=false);
void descKeypoints(float &elapsed_time ,std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, std::string descriptorType);
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType, std::string selectorType);

#endif /* matching2D_hpp */
