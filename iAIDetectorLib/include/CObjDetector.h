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
	// @param[in] cvFrame: input frame in BGR format
	// @return: true if detection is successful, false otherwise. 
	// [Note]: The bounding boxes of detected objects can be obtained by calling GetObjBoxes().
	virtual bool Detect(const cv::Mat& cvFrame) = 0;

	// Set the class names of objects which will be detected by the network
	// @param[in] vClsNames2Detect: class names
	virtual void SetClsNames2Detect(const ObjClsArr& vClsNames2Detect);

	// Draw the bounding boxes on the input frame
	// @param[in] cvFrame: input frame in BGR format
	// @param[in] vBoxes: bounding boxes of detected objects
	// @param[in] bDrawClsName: whether to draw the class name of each object
	// @param[in] bDrawScore: whether to draw the score of each object
	// [Note]: Drawing will be performed on the input frame directly.
	virtual void DrawBBox(cv::Mat& cvFrame, const ObjBoxArr& vBoxes, bool bDrawClsName = true, bool bDrawScore = true);

	// Get the class names of objects which can be detected by the network
	// @return: class names
	const ObjClsArr& GetClsNames() const;

	// Get the bounding boxes of detected objects
	// @return: bounding boxes
	const ObjBoxArr& GetObjBoxes() const;


protected:
	// Perform non-maximum suppression
	// @param[in/out] pvBoxes: bounding boxes of detected objects before and after non-maximum suppression
	inline void NMSBoxes(ObjBoxArr* pvBoxes);

	// Check if the class name is in the list of class names of objects which can be detected by the network
	// @param[in] strClsName: class name
	// @return: true if the class name is in the list, false otherwise
	inline bool IsClsName2Detect(const std::string& strClsName) const;

	// Check if the class ID is in the list of class IDs of objects which can be detected by the network
	// @param[in] nClsID: class ID
	// @return: true if the class ID is in the list, false otherwise
	inline bool IsClsID2Detect(const int nClsID) const;

protected:
	ObjDetNetConfig		m_stObjDetConfig;		// object detection network configuration
	ObjBoxArr			m_vObjBoxes;			// bounding boxes of detected objects
	ObjClsArr			m_vClsNames;			// class names of objects which can be detected by the network
	ObjClsArr			m_vClsNames2Detect;		// class names of objects which will be detected. If empty, all the objects will be detected
};