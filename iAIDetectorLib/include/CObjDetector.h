#pragma once
#include "type_define.h"
#include <opencv2/opencv.hpp>

// Abstract base class for object detection
// All the object detection classes should inherit from this class
class IAIDETECTORLIB_API CObjDetector
{
public:
	CObjDetector(const ObjDetNetConfig& stObjDetNetConfig);
	virtual ~CObjDetector();

	// Detect objects in the input frame
	virtual const ObjBoxArr& Detect(cv::Mat& cvFrame) = 0;


	// Draw the bounding boxes on the input frame
	virtual void DrawBBox(cv::Mat& cvFrame, const ObjBoxArr& vBoxes);

protected:
	// Perform non-maximum suppression
	void NMSBoxes(ObjBoxArr* pvBoxes);

protected:
	ObjDetNetConfig		m_stObjDetConfig;		// object detection network configuration
	ObjBoxArr			m_vObjBoxes;			// bounding boxes of detected objects
	ObjClsArr			m_vClsNames;			// class names of objects which can be detected by the network
};