#pragma once
#include "macro_define.h"
#include <string>

// Structure to hold YOLO network configuration
// The new fields maybe added to this structure in future
typedef struct _YoloNetConfig
{
	float fConfThresh;			// confidence threshold
	float fNMSThresh;			// non-maximum suppression threshold
	std::string sModelPath;		// path to model
	std::string sClassPath;		// path to class names

	_YoloNetConfig()
	{
		fConfThresh = 0.5f;
		fNMSThresh = 0.5f;
		sModelPath = "";
		sClassPath = "";
	}

	_YoloNetConfig(float _fConfThresh = 0.5f, float _fNMSThresh = 0.5f, std::string _sModelPath = "", std::string _sClassPath = "")
	{
		fConfThresh = _fConfThresh;
		fNMSThresh = _fNMSThresh;
		sModelPath = _sModelPath;
		sClassPath = _sClassPath;
	}

}YoloNetConfig;

// Structure to hold bounding box information of detected objects
typedef struct _ObjBBox
{
	float	fX1;		// top left x
	float	fY1;		// top left y
	float	fX2;        // bottom right x
	float	fY2;        // bottom right y
	float	fScore;     // confidence/score
	int		nClassID ;  // predicted class ID. range [0, classes-1]. -1 means invalid

	_ObjBBox()
	{
		fX1 = fY1 = 0.0f;
		fX2 = fY2 = 0.0f;
		fScore = 0.0f;
		nClassID = -1;
	}

	_ObjBBox(float _fX1 = 0.0f, float _fY1 = 0.0f, float _fX2 = 0.0f, float _fY2 = 0.0f, float _fScore = 0.0f, int _nClassID = -1)
	{
		fX1 = _fX1;
		fY1 = _fY1;
		fX2 = _fX2;
		fY2 = _fY2;
		fScore = _fScore;
		nClassID = _nClassID;
	}
} ObjBBox;