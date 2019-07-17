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

using namespace std;

// disable/enable bVis variable in lines 248 , 328 to diable/enable  visualising detectors / matching results

// PROTOTYPE FOR MAIN_FUNCTION
int main_function(string detectorType ,string descriptorType , vector<int> &inRect_keypoints_no_vector,vector<float> &elapsed_time_vector , vector<int> &inRect_matches_no_vector,vector<float> &desc_time_vector  )  ;

/* MAIN PROGRAM */
int main(int argc, const char *argv[])
{    

  	vector<string> detectorTypes ={"SHITOMASI", "HARRIS", "FAST", "BRISK", "ORB", "SIFT" , "AKAZE"};
  	vector<string> descriptorTypes =          { "BRIEF", "FREAK" ,"BRISK", "ORB", "SIFT" , "AKAZE"};
  
  int performance_task = 2; // 2 or 3
 // Performance Task 1 : No of keypoints in all detectors
  if (performance_task ==1){
    std::ofstream printToFile("PerformanceTask1_detectors.txt") ;
    int counter =0;
  	while (counter < detectorTypes.size())
  	{
      	vector<int> inRect_keypoints_no_vector;
        vector<float> elapsed_time_vector;
        vector<int> inRect_matches_no_vector;
        vector<float> desc_time_vector;
        main_function(detectorTypes[counter],descriptorTypes[0],inRect_keypoints_no_vector,elapsed_time_vector,inRect_matches_no_vector,desc_time_vector );   
    	cout << "No of kpts for 10 images in " << detectorTypes[counter] << endl;
    	for(int i=0; i<inRect_keypoints_no_vector.size(); ++i)
    		cout <<  inRect_keypoints_no_vector[i] << " , ";
  		cout << endl;       
        printToFile << "No of kpts for 10 images in " << detectorTypes[counter] << endl;
    	for(int i=0; i<inRect_keypoints_no_vector.size(); ++i)
    		printToFile <<  inRect_keypoints_no_vector[i] << " , ";
  		printToFile << endl;
      	counter +=1;      	
  	}
    printToFile.close();
  }
  
  else 
    {
    std::ofstream printToFile2("PerformanceTask2_matches.txt") ;
	//std::ofstream printToFile3("PerformanceTask3_time.txt") ;
    std::ofstream printToFile3("PerformanceTask3_timeAvg.txt") ;
 	// {"SHITOMASI", "HARRIS", "FAST", "BRISK", "ORB", "SIFT" , "AKAZE"};
  	//             { "BRIEF", "FREAK" ,"BRISK", "ORB", "SIFT" , "AKAZE"};
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
            
            
        printToFile2 << "Compination : " << detectorTypes[i]<< "," << descriptorTypes[j] << endl;
        printToFile3 << detectorTypes[i]<< "," << descriptorTypes[j] << ",";
        for(int c=0; c <= inRect_matches_no_vector.size(); c++)
        {
          cout << c << endl ;
          printToFile2 <<  inRect_matches_no_vector[c] << " , ";
       //   printToFile3 <<  elapsed_time_vector[c] << "," << desc_time_vector[c] << ","  <<  elapsed_time_vector[c] + desc_time_vector[c] << endl ;
          printToFile3 <<  elapsed_time_vector[c] + desc_time_vector[c] << "," ;
        }
        printToFile2 << endl;
        printToFile3 << endl;
      }
    }
  	printToFile2.close();
    printToFile3.close();
    }

}

// main_pipeline of the program that iterates on the images with a given combination of detectorType and descriptorType.
int main_function(string detectorType ,string descriptorType  ,vector<int> &inRect_keypoints_no_vector ,vector<float> &elapsed_time_vector , vector<int> &inRect_matches_no_vector, vector<float> &desc_time_vector   )  
{
    
    /* INIT VARIABLES AND DATA STRUCTURES */
    // data location
    string dataPath = "../";
    // camera
    string imgBasePath = dataPath + "images/";
    string imgPrefix = "KITTI/2011_09_26/image_00/data/000000"; // left camera, color
    string imgFileType = ".png";
    int imgStartIndex = 0; // first file index to load (assumes Lidar and camera names have identical naming convention)
    int imgEndIndex = 9;   // last file index to load
    int imgFillWidth = 4;  // no. of digits which make up the file index (e.g. img-0001.png)

    // misc
    int dataBufferSize = 2;       // no. of images which are held in memory (ring buffer) at the same time
    vector<DataFrame> dataBuffer; // list of data frames which are held in memory at the same time
    bool bVis = false;            // visualize results

//	cout << "OpenCV version : " << CV_VERSION << endl;
  
      /* MAIN LOOP OVER ALL IMAGES */  
    for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex++)
    {
        /* LOAD IMAGE INTO BUFFER */

        // assemble filenames for current index
        ostringstream imgNumber;
        imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
        string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

        // load image from file and convert to grayscale
        cv::Mat img, imgGray;
        img = cv::imread(imgFullFilename);
        cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);

        //// STUDENT ASSIGNMENT
        //// TASK MP.1 -> replace the following code with ring buffer of size dataBufferSize

        // push image into data frame buffer
        DataFrame frame;
        frame.cameraImg = imgGray;
        dataBuffer.push_back(frame);
      	if (imgIndex > dataBufferSize - 1)
		    dataBuffer.erase (dataBuffer.begin());
//       cout << "Size of buffer  (Should always less than 3) = " << dataBuffer.size() << endl;
//       cout << "Image name " <<imgFullFilename << endl ;

        //// EOF STUDENT ASSIGNMENT
        cout << "#1 : LOAD IMAGE INTO BUFFER done" << endl;

        /* DETECT IMAGE KEYPOINTS */

        // extract 2D keypoints from current image
        vector<cv::KeyPoint> keypoints; // create empty feature list for current image

        //// STUDENT ASSIGNMENT
        //// TASK MP.2 -> add the following keypoint detectors in file matching2D.cpp and enable string-based selection based on detectorType
        //// -> HARRIS, FAST, BRISK, ORB, AKAZE, SIFT

      	float elapsed_time = 0.0 ;
        if (detectorType.compare("SHITOMASI") == 0)
        {  	
            detKeypointsShiTomasi(elapsed_time , keypoints, imgGray, false);
       //     cout <<  " Time out of function is : " << elapsed_time <<  endl;
          	elapsed_time_vector.push_back(elapsed_time);
           // keypoints_no_vector.push_back(keypoints.size());
        } 
        else if (detectorType.compare("HARRIS") == 0)
        {
          detKeypointsHarris(elapsed_time , keypoints, imgGray, false);
      //    cout <<  " Time out of function is : " << elapsed_time <<  endl;
          elapsed_time_vector.push_back(elapsed_time);
     //     keypoints_no_vector.push_back(keypoints.size());
        }
        else if (detectorType.compare("FAST") == 0) 
        {
          detKeypointsFAST(elapsed_time ,keypoints, imgGray, false);
      //    cout <<  " Time out of function is : " << elapsed_time <<  endl;
          elapsed_time_vector.push_back(elapsed_time);
     //     keypoints_no_vector.push_back(keypoints.size());
        }
        else if (detectorType.compare("BRISK") == 0)
        {
          detKeypointsBRISK(elapsed_time ,keypoints, imgGray, false);
       //   cout <<  " Time out of function is : " << elapsed_time <<  endl;
          elapsed_time_vector.push_back(elapsed_time);
     //     keypoints_no_vector.push_back(keypoints.size());
        } 
        else if (detectorType.compare("ORB") == 0)
        {
          detKeypointsORB(elapsed_time ,keypoints, imgGray, false);
        //  cout <<  " Time out of function is : " << elapsed_time <<  endl;
          elapsed_time_vector.push_back(elapsed_time);
      //    keypoints_no_vector.push_back(keypoints.size());
        }
        else if (detectorType.compare("AKAZE") == 0)
        {
          detKeypointsAKAZE(elapsed_time ,keypoints, imgGray, false);
        //  cout <<  " Time out of function is : " << elapsed_time <<  endl;
          elapsed_time_vector.push_back(elapsed_time);
      //    keypoints_no_vector.push_back(keypoints.size());
        }
        else if (detectorType.compare("SIFT") == 0)
        {
          detKeypointsSIFT(elapsed_time ,keypoints, imgGray, false);
        //  cout <<  " Time out of function is : " << elapsed_time <<  endl;
          elapsed_time_vector.push_back(elapsed_time);
       //   keypoints_no_vector.push_back(keypoints.size());
        }

     // cout <<  " size SHITOMASI_elapsed_time : " << elapsed_time_vector.size() <<  endl;
        //// EOF STUDENT ASSIGNMENT

        //// STUDENT ASSIGNMENT
        //// TASK MP.3 -> only keep keypoints on the preceding vehicle

        // only keep keypoints on the preceding vehicle
        bool bFocusOnVehicle = true;
        cv::Rect vehicleRect(535, 180, 180, 150);
        if (bFocusOnVehicle)
        {
          vector<cv::KeyPoint> keypointsInsideRect; // create empty feature list for current image
          //cout << " Keypoints[10] "<< keypoints[10].pt << endl;
          int inside  = 0; // no of keypoints inside the rectangle
          int outside = 0; // no of keypoints outside the rectangle
          
          for (int i =0; i < keypoints.size() ; i++){
            
           if (vehicleRect.contains(keypoints[i].pt)){
             keypointsInsideRect.push_back(keypoints[i]);
              inside+=1;            
            }
            else {
              outside+=1;
            }
            
          }
//           cout << " inside : " << inside << " outside : " << outside   << " keypointsBefore size: " << keypoints.size() <<  endl;
          keypoints = keypointsInsideRect; // Take only keypoints inside the Rectangle
          inRect_keypoints_no_vector.push_back(keypoints.size()) ; 
//           cout <<  " keypointsAfter size: " << keypoints.size() <<  endl;
      
        }
      
        bVis = true;
        if (bVis)
    {
       // cout <<  " no keypoints in the preceding vehicle box : " << keypoints.size() <<  endl;
        string windowName = detectorType + " Detection Results: ";
        cv::namedWindow(windowName,1) ; // , 5);
        cv::Mat visImage = imgGray.clone();
        cv::drawKeypoints(imgGray, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        cv::imshow(windowName, visImage);
        cv::waitKey(0);
    }
      bVis = false;

        //// EOF STUDENT ASSIGNMENT

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
        cout << "#2 : DETECT KEYPOINTS done" << endl;

        /* EXTRACT KEYPOINT DESCRIPTORS */

        //// STUDENT ASSIGNMENT
        //// TASK MP.4 -> add the following descriptors in file matching2D.cpp and enable string-based selection based on descriptorType
        //// -> BRIEF, ORB, FREAK, AKAZE, SIFT

        cv::Mat descriptors;
      	float elapsed_time_descriptor = 0.0 ;
                                  
        descKeypoints(elapsed_time_descriptor ,(dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->cameraImg, descriptors, descriptorType);
		desc_time_vector.push_back(elapsed_time_descriptor);
        //// EOF STUDENT ASSIGNMENT

        // push descriptors for current frame to end of data buffer
        (dataBuffer.end() - 1)->descriptors = descriptors;

        cout << "#3 : EXTRACT DESCRIPTORS done" << endl;

        if (dataBuffer.size() > 1) // wait until at least two images have been processed
        {

            /* MATCH KEYPOINT DESCRIPTORS */

            vector<cv::DMatch> matches;
            string matcherType = "MAT_BF";        // MAT_BF, MAT_FLANN
            string binaryOrHogDescriptor = "DES_HOG"; // "DES_BINARY"; // DES_BINARY, DES_HOG
            string selectorType = "SEL_KNN"; // "SEL_NN";       // SEL_NN, SEL_KNN

            //// STUDENT ASSIGNMENT
            //// TASK MP.5 Done -> add FLANN matching in file matching2D.cpp
            //// TASK MP.6 Done -> add KNN match selection and perform descriptor distance ratio filtering with t=0.8 in file matching2D.cpp

            matchDescriptors((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints,
                             (dataBuffer.end() - 2)->descriptors, (dataBuffer.end() - 1)->descriptors,
                             matches, binaryOrHogDescriptor, matcherType, selectorType);


    			inRect_matches_no_vector.push_back(matches.size());

            //// EOF STUDENT ASSIGNMENT

            // store matches in current data frame
            (dataBuffer.end() - 1)->kptMatches = matches;

            cout << "#4 : MATCH KEYPOINT DESCRIPTORS done" << endl;

            // visualize matches between current and previous image
            bVis = true;
            if (bVis)
            {
                cv::Mat matchImg = ((dataBuffer.end() - 1)->cameraImg).clone();
                cv::drawMatches((dataBuffer.end() - 2)->cameraImg, (dataBuffer.end() - 2)->keypoints,
                                (dataBuffer.end() - 1)->cameraImg, (dataBuffer.end() - 1)->keypoints,
                                matches, matchImg,
                                cv::Scalar::all(-1), cv::Scalar::all(-1),
                                vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

                string windowName = "Matching keypoints " +descriptorType   ;
                cv::namedWindow(windowName , 7 );
                cv::imshow(windowName, matchImg);
                cout << "Press key to continue to next image" << endl;
                cv::waitKey(0); // wait for key to be pressed
            }
            bVis = false;
        }

    } // eof loop over all images
 

    return 0;
}
