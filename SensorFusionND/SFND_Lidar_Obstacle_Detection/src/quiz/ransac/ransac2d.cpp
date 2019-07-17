/* \author Aaron Brown */
// Quiz on implementing simple RANSAC line fitting

#include "../../render/render.h"
#include <unordered_set>
#include "../../processPointClouds.h"
// using templates for processPointClouds so also include .cpp to help linker
#include "../../processPointClouds.cpp"

pcl::PointCloud<pcl::PointXYZ>::Ptr CreateData()
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
  	// Add inliers
  	float scatter = 0.6;
  	for(int i = -5; i < 5; i++)
  	{
  		double rx = 2*(((double) rand() / (RAND_MAX))-0.5);
  		double ry = 2*(((double) rand() / (RAND_MAX))-0.5);
  		pcl::PointXYZ point;
  		point.x = i+scatter*rx;
  		point.y = i+scatter*ry;
  		point.z = 0;

  		cloud->points.push_back(point);
  	}
  	// Add outliers
  	int numOutliers = 10;
  	while(numOutliers--)
  	{
  		double rx = 2*(((double) rand() / (RAND_MAX))-0.5);
  		double ry = 2*(((double) rand() / (RAND_MAX))-0.5);
  		pcl::PointXYZ point;
  		point.x = 5*rx;
  		point.y = 5*ry;
  		point.z = 0;

  		cloud->points.push_back(point);

  	}
  	cloud->width = cloud->points.size();
  	cloud->height = 1;

  	return cloud;

}

pcl::PointCloud<pcl::PointXYZ>::Ptr CreateData3D()
{
	ProcessPointClouds<pcl::PointXYZ> pointProcessor;
	return pointProcessor.loadPcd("../../../sensors/data/pcd/simpleHighway.pcd");
}


pcl::visualization::PCLVisualizer::Ptr initScene()
{
	pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer ("2D Viewer"));
	viewer->setBackgroundColor (0, 0, 0);
  	viewer->initCameraParameters();
  	viewer->setCameraPosition(0, 0, 15, 0, 1, 0);
  	viewer->addCoordinateSystem (1.0);
  	return viewer;
}

std::unordered_set<int> Ransac(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, int maxIterations, float distanceTol)
{
	std::unordered_set<int> inliersResult;
	srand(time(NULL));
	
	// TODO: Fill in this function

	// For max iterations 

	// Randomly sample subset and fit line

	// Measure distance between every point and fitted line
	// If distance is smaller than threshold count it as inlier

	// Return indicies of inliers from fitted line with most inliers
		// Return indicies of inliers from fitted line with most inliers
	while (maxIterations--)
	{
	    // Randomly sample subset and fit line
	    std::unordered_set<int> inliers;
	    
	    while (inliers.size() < 2)
	        inliers.insert(rand()%(cloud->points.size()));
        
        float x1, x2, y1, y2;
        
		// point 1
        auto itr = inliers.begin();
        x1 = cloud->points[*itr].x;
        y1 = cloud->points[*itr].y;
        // point 2
		itr++;
        x2 = cloud->points[*itr].x;
        y2 = cloud->points[*itr].y;
        // line equ.
        float a = y1 - y2;
        float b = x2 - x1;
        float c = x1*y2 - x2*y1 ;
        
        for (int idx = 0; idx < cloud->points.size(); idx++)
        {
            if(inliers.count(idx) > 0)
                continue;
            // point under test
            pcl::PointXYZ point = cloud->points[idx];
            float x3 = point.x;
            float y3 = point.y;
            // find distance to the point
            float d = fabs(a*x3+b*y3+c)/sqrt(a*a+b*b);
            // check distance 
            if (d <= distanceTol)
                inliers.insert(idx); 
        }
        
        if (inliers.size() > inliersResult.size())
            inliersResult = inliers;
    }
    
    auto endTime = std::chrono::steady_clock::now(); 
	auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
	std::cout << "Ransan Time: " << elapsedTime.count() << " ms" << std::endl;
  
	return inliersResult;

}

std::unordered_set<int> Ransac3D(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, int maxIterations, float distanceTol)
{
    auto startTime = std::chrono::steady_clock::now();
    
    std::unordered_set<int> inliersResult;
    srand(time(NULL));
    
    // TODO: Fill in this function
    
    int best_a, best_b, best_c, best_d;
    int max_inliners = 0;
    int size = cloud->points.size();
    
    // For max iterations
	while (maxIterations--)
    {
	    std::unordered_set<int> inliers;
	    int inliners = 0;
        // Randomly sample subset and fit line
	    while (inliers.size() < 3)
	        inliers.insert(rand()%(cloud->points.size()));
	        
        float x1, x2, x3, y1, y2, y3, z1, z2, z3;
        auto itr = inliers.begin();
        x1 = cloud->points[*itr].x;
        y1 = cloud->points[*itr].y;
        z1 = cloud->points[*itr].z;
        itr++;
        x2 = cloud->points[*itr].x;
        y2 = cloud->points[*itr].y;
        z2 = cloud->points[*itr].z;
        itr++;
        x3 = cloud->points[*itr].x;
        y3 = cloud->points[*itr].y;
        z3 = cloud->points[*itr].z;


        float a = (y2-y1)*(z3-z1) - (z2-z1)*(y3-y1);
        
        float b = - (x2-x1)*(z3-z1) + (z2-z1)*(x3-x1);
        
        float c = (x2-x1)*(y3-y1) - (y2-y1)*(x3-x1);
        
        float d = -(a*x1 + b*y1 + c*z1);

//////////////////////////////////////////////////////////////////
/*
        int ind1 = std::get<0>(ind);
        int ind2 = std::get<1>(ind);
        int ind3 = std::get<2>(ind);

        float a = (cloud->points[ind2].y-cloud->points[ind1].y)*(cloud->points[ind3].z-cloud->points[ind1].z)
        - (cloud->points[ind2].z-cloud->points[ind1].z)*(cloud->points[ind3].y-cloud->points[ind1].y);
        
        float b = - (cloud->points[ind2].x-cloud->points[ind1].x)*(cloud->points[ind3].z-cloud->points[ind1].z)
        + (cloud->points[ind2].z-cloud->points[ind1].z)*(cloud->points[ind3].x-cloud->points[ind1].x);
        
        float c = (cloud->points[ind2].x-cloud->points[ind1].x)*(cloud->points[ind3].y-cloud->points[ind1].y)
        - (cloud->points[ind2].y-cloud->points[ind1].y)*(cloud->points[ind3].x-cloud->points[ind1].x);

        float x = cloud->points[ind1].x;
        float y = cloud->points[ind1].y;
        float z = cloud->points[ind1].z;
        
        float d = - (a*x + b*y + c*z);
*/
/////////////////////////////////////////////////////////////////        
        float div = sqrt(a*a + b*b + c*c);
                    // Measure distance between every point and fitted line
        for(int j=0;j<size;j++)
        {
            float dist = fabs(cloud->points[j].x*a + cloud->points[j].y*b + cloud->points[j].z*c + d)/div;
            // If distance is smaller than threshold count it as inlier
            if(dist <= distanceTol)
                inliners++;
        }
        if(inliners > max_inliners)
        {
            best_a = a;
            best_b = b;
            best_c = c;
            best_d = d;
            max_inliners = inliners;
        }
    }
    
    // Return indicies of inliers from fitted line with most inliers
    for(int j=0;j<size;j++)
    {
        float div = sqrt(best_a*best_a + best_b*best_b + best_c*best_c);
        float dist = fabs(cloud->points[j].x*best_a + cloud->points[j].y*best_b + cloud->points[j].z*best_c + best_d)/div;
        // If distance is smaller than threshold count it as inlier
        if(dist <= distanceTol)
            inliersResult.insert(j);
    }
    
    auto endTime = std::chrono::steady_clock::now();
    auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cout << "plane segmentation took " << elapsedTime.count() << " milliseconds" << std::endl;
    
    return inliersResult;
}

int main ()
{

	// Create viewer
	pcl::visualization::PCLVisualizer::Ptr viewer = initScene();

	// Create data
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud = CreateData3D();
	

	// TODO: Change the max iteration and distance tolerance arguments for Ransac function
	std::unordered_set<int> inliers = Ransac(cloud, 0, 0);

	pcl::PointCloud<pcl::PointXYZ>::Ptr  cloudInliers(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloudOutliers(new pcl::PointCloud<pcl::PointXYZ>());

	for(int index = 0; index < cloud->points.size(); index++)
	{
		pcl::PointXYZ point = cloud->points[index];
		if(inliers.count(index))
			cloudInliers->points.push_back(point);
		else
			cloudOutliers->points.push_back(point);
	}


	// Render 2D point cloud with inliers and outliers
	if(inliers.size())
	{
		renderPointCloud(viewer,cloudInliers,"inliers",Color(0,1,0));
  		renderPointCloud(viewer,cloudOutliers,"outliers",Color(1,0,0));
	}
  	else
  	{
  		renderPointCloud(viewer,cloud,"data");
  	}
	
  	while (!viewer->wasStopped ())
  	{
  	  viewer->spinOnce ();
  	}
  	
}
