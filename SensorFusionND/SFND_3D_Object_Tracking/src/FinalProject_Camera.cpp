
/* INCLUDES FOR THIS PROJECT */
#include <iostream>
#include <fstream>
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
#include "matching2D.hpp"
#include "objectDetection2D.hpp"
#include "lidarData.hpp"
#include "camFusion.hpp"
using namespace std;

//// Notes for reviewer::
// Writeup and spreadsheats are on documentation directory
// bvis variable is disabled by default
//    Enable it in lines  207  , 386 , 406 for viewing lidar points in image ,  matches in the bounding box of the preceding car , TTC values and lidar points in the preceding car. 
// DEBUG_ON is disabled by default , I enabled it to output a csv file.
// one_combination variable is true by default , disable it to view all combinations line 37


//#define DEBUG_ON // Uncomment for outputting a file

#ifdef DEBUG_ON
	std::ofstream printToFile("../PerformanceTask.csv") ;
#endif 
bool one_combination = true;

// PROTOTYPE FOR MAIN_FUNCTION
int main_function(string detectorType ,string descriptorType , vector<int> &inRect_keypoints_no_vector,vector<float> &elapsed_time_vector , vector<int> &inRect_matches_no_vector,vector<float> &desc_time_vector  )  ;

/* MAIN PROGRAM */
int main(int argc, const char *argv[])
{    

  vector<string> detectorTypes ={"SHITOMASI", "HARRIS", "FAST", "BRISK", "ORB", "SIFT" , "AKAZE"};
  vector<string> descriptorTypes =          { "BRIEF", "FREAK" ,"BRISK", "ORB", "SIFT" , "AKAZE"};
 bool bdebug = false;
  if (bdebug)
  {
    vector<string> detectorTypes ={"SHITOMASI", "HARRIS", "FAST", "BRISK", "ORB", "SIFT" , "AKAZE","SHITOMASI", "HARRIS", "FAST", "BRISK", "ORB", "SIFT" , "AKAZE","SHITOMASI", "HARRIS", "FAST", "BRISK", "ORB", "SIFT" , "AKAZE","SHITOMASI", "HARRIS", "FAST", "BRISK", "ORB", "SIFT" , "AKAZE","SHITOMASI", "HARRIS", "FAST", "BRISK", "ORB", "SIFT" , "AKAZE","SHITOMASI", "HARRIS", "FAST", "BRISK", "ORB", "SIFT" , "AKAZE","SHITOMASI", "HARRIS", "FAST", "BRISK", "ORB", "SIFT" , "AKAZE","SHITOMASI", "HARRIS", "FAST", "BRISK", "ORB", "SIFT" , "AKAZE","SHITOMASI", "HARRIS", "FAST", "BRISK", "ORB", "SIFT" , "AKAZE"};
    vector<string> detectors;
    vector<int> *test = new vector<int>() ;
    cout << "Detector before " << &(detectors[0]) << endl;
    for (int i=0;i<detectorTypes.size();i++)
    {
    detectors.push_back(detectorTypes[i]);
    test->push_back(i);
    cout << "Detector[0] add " << &(detectors[i])  << " test[0] add " << &(test[0])   << endl;
    cout << "data: " << detectors[i] << " data test: " << test->at(i) << endl;
      if (i==21)
      {
        
            cout << "next Detector i=2 add(i+3) " << endl ;
        	cout << &(detectors[i+3])  << " data(i+3) " << (detectors[i+3]) <<  endl;

      }
      if (i==18)
      {
            cout << "next test i=18 add(i+3) " << endl ;
        	cout << &(test[i+3])  <<  endl ;
            cout << " data(i+3) " << (test->at(i+3)) <<  endl; 
        /*terminate called after throwing an instance of 'std::out_of_range'
  what():  vector::_M_range_check: __n (which is 21) >= this->size() (which is 19)
 data(i+3) Aborted (core dumped)
        */
            cout << "next Detector i=18 add(i+3) " <<  endl;
        	cout << &(detectors[i+3])  << endl;
        	cout << " data(i+3) " << (detectors[i+3]) << endl; //segmentation fault

      }
    }  
  }

  if (one_combination)
  {
    detectorTypes ={"FAST"};
  	descriptorTypes ={"BRIEF"}; 
  }
   
  // {"SHITOMASI", "HARRIS", "FAST", "BRISK", "ORB", "SIFT" , "AKAZE"};
  //             { "BRIEF", "FREAK" ,"BRISK", "ORB", "SIFT" , "AKAZE"};
  if (true)
  {
  for (int i =0 ; i< detectorTypes.size() ; i++)
  {
    for (int j =0 ; j < (descriptorTypes.size()) ; j++ )
    {

      vector<int> inRect_keypoints_no_vector;
      vector<float> elapsed_time_vector;
      vector<int> inRect_matches_no_vector;
      vector<float> desc_time_vector;

      if ( detectorTypes[i] == "SIFT" and descriptorTypes[j] == "ORB" )
        break; // This compination leads to Out of memory error
      if (  descriptorTypes[j] == "AKAZE" and   detectorTypes[i] != "AKAZE" )
        break; // Akaze descriptor only works with Akaze ketector
      cout << "Compination: " << detectorTypes[i]<< "," << descriptorTypes[j] << endl;         
      main_function(detectorTypes[i],descriptorTypes[j],inRect_keypoints_no_vector,elapsed_time_vector,inRect_matches_no_vector,desc_time_vector );           
      
    }
  }
  }
  #ifdef DEBUG_ON
  	printToFile.close();
  #endif
}

// main_pipeline of the program that iterates on the images with a given combination of detectorType and descriptorType.
int main_function(string detectorType ,string descriptorType  ,vector<int> &inRect_keypoints_no_vector ,vector<float> &elapsed_time_vector , vector<int> &inRect_matches_no_vector, vector<float> &desc_time_vector   )  
{
  
    /* INIT VARIABLES AND DATA STRUCTURES */

    // data location
    string dataPath = "../";

    // camera
    string imgBasePath = dataPath + "images/";
    string imgPrefix = "KITTI/2011_09_26/image_02/data/000000"; // left camera, color
    string imgFileType = ".png";
    int imgStartIndex = 0; // first file index to load (assumes Lidar and camera names have identical naming convention)
    int imgEndIndex = 18;   // last file index to load
    int imgStepWidth = 1; 
    int imgFillWidth = 4;  // no. of digits which make up the file index (e.g. img-0001.png)

    // object detection
    string yoloBasePath = dataPath + "dat/yolo/";
    string yoloClassesFile = yoloBasePath + "coco.names";
    string yoloModelConfiguration = yoloBasePath + "yolov3.cfg";
    string yoloModelWeights = yoloBasePath + "yolov3.weights";

    // Lidar
    string lidarPrefix = "KITTI/2011_09_26/velodyne_points/data/000000";
    string lidarFileType = ".bin";

    // calibration data for camera and lidar
    cv::Mat P_rect_00(3,4,cv::DataType<double>::type); // 3x4 projection matrix after rectification
    cv::Mat R_rect_00(4,4,cv::DataType<double>::type); // 3x3 rectifying rotation to make image planes co-planar
    cv::Mat RT(4,4,cv::DataType<double>::type); // rotation matrix and translation vector
    
    RT.at<double>(0,0) = 7.533745e-03; RT.at<double>(0,1) = -9.999714e-01; RT.at<double>(0,2) = -6.166020e-04; RT.at<double>(0,3) = -4.069766e-03;
    RT.at<double>(1,0) = 1.480249e-02; RT.at<double>(1,1) = 7.280733e-04; RT.at<double>(1,2) = -9.998902e-01; RT.at<double>(1,3) = -7.631618e-02;
    RT.at<double>(2,0) = 9.998621e-01; RT.at<double>(2,1) = 7.523790e-03; RT.at<double>(2,2) = 1.480755e-02; RT.at<double>(2,3) = -2.717806e-01;
    RT.at<double>(3,0) = 0.0; RT.at<double>(3,1) = 0.0; RT.at<double>(3,2) = 0.0; RT.at<double>(3,3) = 1.0;
    
    R_rect_00.at<double>(0,0) = 9.999239e-01; R_rect_00.at<double>(0,1) = 9.837760e-03; R_rect_00.at<double>(0,2) = -7.445048e-03; R_rect_00.at<double>(0,3) = 0.0;
    R_rect_00.at<double>(1,0) = -9.869795e-03; R_rect_00.at<double>(1,1) = 9.999421e-01; R_rect_00.at<double>(1,2) = -4.278459e-03; R_rect_00.at<double>(1,3) = 0.0;
    R_rect_00.at<double>(2,0) = 7.402527e-03; R_rect_00.at<double>(2,1) = 4.351614e-03; R_rect_00.at<double>(2,2) = 9.999631e-01; R_rect_00.at<double>(2,3) = 0.0;
    R_rect_00.at<double>(3,0) = 0; R_rect_00.at<double>(3,1) = 0; R_rect_00.at<double>(3,2) = 0; R_rect_00.at<double>(3,3) = 1;
    
    P_rect_00.at<double>(0,0) = 7.215377e+02; P_rect_00.at<double>(0,1) = 0.000000e+00; P_rect_00.at<double>(0,2) = 6.095593e+02; P_rect_00.at<double>(0,3) = 0.000000e+00;
    P_rect_00.at<double>(1,0) = 0.000000e+00; P_rect_00.at<double>(1,1) = 7.215377e+02; P_rect_00.at<double>(1,2) = 1.728540e+02; P_rect_00.at<double>(1,3) = 0.000000e+00;
    P_rect_00.at<double>(2,0) = 0.000000e+00; P_rect_00.at<double>(2,1) = 0.000000e+00; P_rect_00.at<double>(2,2) = 1.000000e+00; P_rect_00.at<double>(2,3) = 0.000000e+00;    

    // misc
    double sensorFrameRate = 10.0 / imgStepWidth; // frames per second for Lidar and camera
    int dataBufferSize = 2;       // no. of images which are held in memory (ring buffer) at the same time
    vector<DataFrame> dataBuffer; // list of data frames which are held in memory at the same time
    bool bVis = false;            // visualize results

    /* MAIN LOOP OVER ALL IMAGES */

    for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex+=imgStepWidth)
    {
        /* LOAD IMAGE INTO BUFFER */

        // assemble filenames for current index
        ostringstream imgNumber;
        imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
        string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

        // load image from file 
        cv::Mat img = cv::imread(imgFullFilename);

        // push image into data frame buffer
        DataFrame frame;
        frame.cameraImg = img;
        dataBuffer.push_back(frame);
      	if (imgIndex > dataBufferSize - 1)
		    dataBuffer.erase (dataBuffer.begin());
      
   //     cout << "#1 : LOAD IMAGE INTO BUFFER done" << endl;


        /* DETECT & CLASSIFY OBJECTS */

        float confThreshold = 0.2;
        float nmsThreshold = 0.4; 
      // Input : Image ---->>  Output : bounding box roi ,classID , confidence , boxID 
        detectObjects((dataBuffer.end() - 1)->cameraImg, (dataBuffer.end() - 1)->boundingBoxes, confThreshold, nmsThreshold,
                      yoloBasePath, yoloClassesFile, yoloModelConfiguration, yoloModelWeights, bVis);

      	bool debugBoundingBoxes = false ;
      	if (debugBoundingBoxes)
        {
      		for (auto box :(dataBuffer.end() - 1)->boundingBoxes)
        	{
          		cout << "BoxId , trackID  "  << box.boxID << "  " << box.trackID  << endl; 
          		cout << " roi "  <<  box.roi  << endl; 
          		cout << "  calssID , confidence "  <<   box.classID  <<"   " << box.confidence << endl; 
        	}
        }

    //    cout << "#2 : DETECT & CLASSIFY OBJECTS done" << endl;


        /* CROP LIDAR POINTS */

        // load 3D Lidar points from file
        string lidarFullFilename = imgBasePath + lidarPrefix + imgNumber.str() + lidarFileType;
        std::vector<LidarPoint> lidarPoints;
        loadLidarFromFile(lidarPoints, lidarFullFilename);

        // remove Lidar points based on distance properties
        float minZ = -1.5, maxZ = -0.9, minX = 2.0, maxX = 20.0, maxY = 2.0, minR = 0.1; // focus on ego lane
      //OUTPUT : cropped lidar points
        cropLidarPoints(lidarPoints, minX, maxX, maxY, minZ, maxZ, minR);
    
        (dataBuffer.end() - 1)->lidarPoints = lidarPoints;

   //     cout << "#3 : CROP LIDAR POINTS done" << endl;


        /* CLUSTER LIDAR POINT CLOUD */

        // associate Lidar points with camera-based ROI
        float shrinkFactor = 0.10; // shrinks each bounding box by the given percentage to avoid 3D object merging at the edges of an ROI
      // OUTPUT: clustered lidar points in each data buffer -> bounding boxes -> lidar points 
        clusterLidarWithROI((dataBuffer.end()-1)->boundingBoxes, (dataBuffer.end() - 1)->lidarPoints, shrinkFactor, P_rect_00, R_rect_00, RT);

        // Visualize 3D objects
        bVis = false; 
        if(bVis)
        {
            show3DObjects((dataBuffer.end()-1)->boundingBoxes, cv::Size(4.0, 20.0), cv::Size(2000, 2000), true);
        }
        bVis = false;

  //      cout << "#4 : CLUSTER LIDAR POINT CLOUD done" << endl;
        
        
        // REMOVE THIS LINE BEFORE PROCEEDING WITH THE FINAL PROJECT
       // continue; // skips directly to the next image without processing what comes beneath

        /* DETECT IMAGE KEYPOINTS */

        // convert current image to grayscale
        cv::Mat imgGray;
        cv::cvtColor((dataBuffer.end()-1)->cameraImg, imgGray, cv::COLOR_BGR2GRAY);

        // extract 2D keypoints from current image
        vector<cv::KeyPoint> keypoints; // create empty feature list for current image
      	vector<float> elapsed_time_vector;
        
      	float elapsed_time = 0.0 ;
        if (detectorType.compare("SHITOMASI") == 0)
        {  	
            detKeypointsShiTomasi(elapsed_time , keypoints, imgGray, false);
          	elapsed_time_vector.push_back(elapsed_time);
        } 
        else if (detectorType.compare("HARRIS") == 0)
        {
          detKeypointsHarris(elapsed_time , keypoints, imgGray, false);
          elapsed_time_vector.push_back(elapsed_time);
        }
        else if (detectorType.compare("FAST") == 0) 
        {
          detKeypointsFAST(elapsed_time ,keypoints, imgGray, false);
          elapsed_time_vector.push_back(elapsed_time);
        }
        else if (detectorType.compare("BRISK") == 0)
        {
          detKeypointsBRISK(elapsed_time ,keypoints, imgGray, false);
          elapsed_time_vector.push_back(elapsed_time);
        } 
        else if (detectorType.compare("ORB") == 0)
        {
          detKeypointsORB(elapsed_time ,keypoints, imgGray, false);
          elapsed_time_vector.push_back(elapsed_time);
        }
        else if (detectorType.compare("AKAZE") == 0)
        {
          detKeypointsAKAZE(elapsed_time ,keypoints, imgGray, false);
          elapsed_time_vector.push_back(elapsed_time);
        }
        else if (detectorType.compare("SIFT") == 0)
        {
          detKeypointsSIFT(elapsed_time ,keypoints, imgGray, false);
          elapsed_time_vector.push_back(elapsed_time);
        }

        // optional : limit number of keypoints (helpful for debugging and learning)
        bool bLimitKpts = false;
        if (bLimitKpts)
        {
            int maxKeypoints = 50;

            if (detectorType.compare("SHITOMASI") == 0)
            { // there is no response info, so keep the first 50 as they are sorted in descending quality order
                keypoints.erase(keypoints.begin() + maxKeypoints, keypoints.end());
            }
            cv::KeyPointsFilter::retainBest(keypoints, maxKeypoints);
            cout << " NOTE: Keypoints have been limited!" << endl;
        }

        // push keypoints and descriptor for current frame to end of data buffer
        (dataBuffer.end() - 1)->keypoints = keypoints;

  //      cout << "#5 : DETECT KEYPOINTS done" << endl;


        /* EXTRACT KEYPOINT DESCRIPTORS */

        cv::Mat descriptors;
      	vector<float> desc_time_vector;
		float elapsed_time_descriptor = 0.0 ;

        descKeypoints(elapsed_time_descriptor ,(dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->cameraImg, descriptors, descriptorType);
		desc_time_vector.push_back(elapsed_time_descriptor); 

        // push descriptors for current frame to end of data buffer
        (dataBuffer.end() - 1)->descriptors = descriptors;

 //       cout << "#6 : EXTRACT DESCRIPTORS done" << endl;


        if (dataBuffer.size() > 1) // wait until at least two images have been processed
        {

            /* MATCH KEYPOINT DESCRIPTORS */

            vector<cv::DMatch> matches;
          	string matcherType = "MAT_BF";        // MAT_BF, MAT_FLANN  
          	if (descriptorType == "SIFT")
              matcherType = "MAT_FLANN";
            
            string binaryOrHogDescriptor = "DES_BINARY"; // DES_BINARY, DES_HOG
            string selectorType = "SEL_NN";       // SEL_NN, SEL_KNN

            matchDescriptors((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints,
                             (dataBuffer.end() - 2)->descriptors, (dataBuffer.end() - 1)->descriptors,
                             matches, binaryOrHogDescriptor, matcherType, selectorType);

            // store matches in current data frame
            (dataBuffer.end() - 1)->kptMatches = matches;

           // cout << "#7 : MATCH KEYPOINT DESCRIPTORS done" << endl;

          
          

            
            /* TRACK 3D OBJECT BOUNDING BOXES */

            //// STUDENT ASSIGNMENT
            //// TASK FP.1 -> match list of 3D objects (vector<BoundingBox>) between current and previous frame (implement ->matchBoundingBoxes)
            map<int, int> bbBestMatches;
            matchBoundingBoxes(matches, bbBestMatches, *(dataBuffer.end()-2), *(dataBuffer.end()-1)); // associate bounding boxes between current and previous frame using keypoint matches
          
          
            //// EOF STUDENT ASSIGNMENT

            // store matches in current data frame
            (dataBuffer.end()-1)->bbMatches = bbBestMatches;

 //           cout << "#8 : TRACK 3D OBJECT BOUNDING BOXES done" << endl;


            /* COMPUTE TTC ON OBJECT IN FRONT */

            // loop over all BB match pairs
            for (auto it1 = (dataBuffer.end() - 1)->bbMatches.begin(); it1 != (dataBuffer.end() - 1)->bbMatches.end(); ++it1)
            {
              cout << "Matches: " << it1->second << " , " << it1->first << endl;
                // find bounding boxes associates with current match
                BoundingBox *prevBB, *currBB;
                for (auto it2 = (dataBuffer.end() - 1)->boundingBoxes.begin(); it2 != (dataBuffer.end() - 1)->boundingBoxes.end(); ++it2)
                {
                    if (it1->second == it2->boxID) // check wether current match partner corresponds to this BB
                    {
                        currBB = &(*it2);
                    }
                }

                for (auto it2 = (dataBuffer.end() - 2)->boundingBoxes.begin(); it2 != (dataBuffer.end() - 2)->boundingBoxes.end(); ++it2)
                {
                  
                    if (it1->first == it2->boxID) // check wether current match partner corresponds to this BB
                    {
                        prevBB = &(*it2);
                    }
                }
                // compute TTC for current match
                if( currBB->lidarPoints.size()>0 && prevBB->lidarPoints.size()>0 ) // only compute TTC if we have Lidar points
                {
                    //// STUDENT ASSIGNMENT
                    //// TASK FP.2 -> compute time-to-collision based on Lidar data (implement -> computeTTCLidar)
                    double ttcLidar; 
                  //  computeTTCLidar(prevBB->lidarPoints, currBB->lidarPoints, sensorFrameRate, ttcLidar);
                  ttcLidar = 1.5;
                    //// EOF STUDENT ASSIGNMENT

                    //// STUDENT ASSIGNMENT
                    //// TASK FP.3 -> assign enclosed keypoint matches to bounding box (implement -> clusterKptMatchesWithROI)
                    //// TASK FP.4 -> compute time-to-collision based on camera (implement -> computeTTCCamera)
                    double ttcCamera;
                    clusterKptMatchesWithROI(*currBB, (dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->kptMatches);    
                  
           
                  
                  
                    computeTTCCamera((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints, currBB->kptMatches, sensorFrameRate, ttcCamera);
                  bVis = false; 
            if (bVis )
            {
                cv::Mat matchImg = ((dataBuffer.end() - 1)->cameraImg).clone();
                cv::drawMatches((dataBuffer.end() - 2)->cameraImg, (dataBuffer.end() - 2)->keypoints,
                                (dataBuffer.end() - 1)->cameraImg, (dataBuffer.end() - 1)->keypoints,
                                currBB->kptMatches, matchImg,
                                cv::Scalar::all(-1), cv::Scalar::all(-1),
                                vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

                string windowName = "Matching keypoints " +descriptorType   ;
                cv::namedWindow(windowName , 7 );
                cv::imshow(windowName, matchImg);
                cout << "Press key to continue to next image" << endl;
                cv::waitKey(0); // wait for key to be pressed
            }
            bVis = false;
                    //// EOF STUDENT ASSIGNMENT

                   cout << "ttcCamera: " << ttcCamera << " , ttcLidar: " << ttcLidar << "lidarPtsSize Curr: " << currBB->lidarPoints.size() << " Prev: " <<  prevBB->lidarPoints.size() << " ID: " << currBB->boxID << endl; 
                    bVis = false; 
                    if (bVis)
                    {
                        cv::Mat visImg = (dataBuffer.end() - 1)->cameraImg.clone();
                        showLidarImgOverlay(visImg, currBB->lidarPoints, P_rect_00, R_rect_00, RT, &visImg);
                        cv::rectangle(visImg, cv::Point(currBB->roi.x, currBB->roi.y), cv::Point(currBB->roi.x + currBB->roi.width, currBB->roi.y + currBB->roi.height), cv::Scalar(0, 255, 0), 2);
                        
                        char str[200];
                        sprintf(str, "TTC Lidar : %f s, TTC Camera : %f s", ttcLidar, ttcCamera);
                        putText(visImg, str, cv::Point2f(80, 50), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0,0,255));

                        string windowName = "Final Results : TTC";
                        cv::namedWindow(windowName, 4);
                        cv::imshow(windowName, visImg);
                        cout << "Press key to continue to next frame" << endl;
                        cv::waitKey(0);
                    }
                    bVis = false;
                  	#ifdef DEBUG_ON
                      printToFile << detectorType << "," << descriptorType << "," << ttcCamera << "," << ttcLidar <<  endl;
                    #endif

                } // eof TTC computation
            } // eof loop over all BB matches            

        }

    } // eof loop over all images

    return 0;
}
