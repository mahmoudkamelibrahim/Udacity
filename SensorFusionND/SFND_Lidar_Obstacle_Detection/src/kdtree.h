// PCL lib Functions for processing point clouds 

#ifndef KDTREE_H_
#define KDTREE_H_

#include <pcl/io/pcd_io.h>
#include <pcl/common/common.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/crop_box.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/common/transforms.h>
#include <iostream> 
#include <string>  
#include <vector>
#include <ctime>
#include <chrono>
#include "render/box.h"

// Structure to represent node of kd tree
struct Node
{
	pcl::PointXYZI point;
	int id;
	Node* left;
	Node* right;

	Node(pcl::PointXYZI arr, int setId)
	:	point(arr), id(setId), left(NULL), right(NULL)
	{}
};

struct KdTree
{
	Node* root;

	KdTree()
	: root(NULL)
	{}

	void insertHelper(Node** node, int depth, pcl::PointXYZI target, int id)
    {
      if(*node == NULL) 
        *node = new Node(target, id);
      else
      {
        uint cd = depth % 3;
        if(cd == 0)
        {
            if(target.x < (*node)->point.x)
                insertHelper(&(*node)->left, depth + 1, target, id);
            else
                insertHelper(&(*node)->right, depth + 1, target, id);
        }
        else if (cd == 1)
        {
            if(target.y < (*node)->point.y)
                insertHelper(&(*node)->left, depth + 1, target, id);
            else
                insertHelper(&(*node)->right, depth + 1, target, id);
        }
        else
        {
            if(target.z < (*node)->point.z)
                insertHelper(&(*node)->left, depth + 1, target, id);
            else
                insertHelper(&(*node)->right, depth + 1, target, id);
        }
        
      }
    }
  
void insert(pcl::PointXYZI point, int id)
{
	// TODO: Fill in this function to insert a new point into the tree
	// the function should create a new node and place correctly with in the root 
	insertHelper(&root, 0, point, id);
}

void searchHelper(pcl::PointXYZI target, Node* node, int depth, float distanceTol, std::vector<int>& ids)
{
	if(node != NULL)
    {
        if((node->point.x >= (target.x-distanceTol)&&(node->point.x <= (target.x+distanceTol))) && (node->point.y >= (target.y-distanceTol)&&(node->point.y <= (target.y+distanceTol))) && (node->point.z >= (target.z-distanceTol) && (node->point.z <= (target.z+distanceTol)) ))
        {
            float distance = sqrt((node->point.x - target.x) * (node->point.x - target.x) + (node->point.y - target.y) * (node->point.y - target.y) + (node->point.z - target.z) * (node->point.z - target.z));
            if(distance <= distanceTol)
                ids.push_back(node->id);
        }

        uint cd = depth % 3;
        if(cd == 0)
        {
            if((target.x - distanceTol) < node->point.x)
                searchHelper(target, node->left, depth+1, distanceTol, ids);
            if((target.x + distanceTol) > node->point.x) // Must be if (else and else if don't work)
                searchHelper(target, node->right, depth+1, distanceTol, ids);
        }
        else if (cd == 1)
        {
            if((target.y - distanceTol) < node->point.y)
                searchHelper(target, node->left, depth+1, distanceTol, ids);
            if((target.y + distanceTol) > node->point.y)
                searchHelper(target, node->right, depth+1, distanceTol, ids);
        }
        else
        {
            if((target.z - distanceTol) < node->point.z)
                searchHelper(target, node->left, depth+1, distanceTol, ids);
            if((target.z + distanceTol) > node->point.z)
                searchHelper(target, node->right, depth+1, distanceTol, ids);
        }
    }
}
  
	// return a list of point ids in the tree that are within distance of target
	std::vector<int> search(pcl::PointXYZI target, float distanceTol)
	{
		std::vector<int> ids;
        searchHelper(target, root, 0, distanceTol, ids);
		return ids;
	}
	

};
#endif /* PROCESSPOINTCLOUDS_H_ */