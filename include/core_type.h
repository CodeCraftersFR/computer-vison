#pragma once
#include <vector>
#include <string>
//########################################################################
// Common data structures
//########################################################################

// Structure to hold the detailed configuration information of all types of deep learning networks
// The new fields maybe added to this structure in future
typedef struct _NetDetailsConfig {
	int		nDeviceID;		// -1: CPU, >=0: GPU device ID
	double	dNormMean0; 	// normalisation mean value for the 1st channel
	double	dNormMean1; 	// normalisation mean value for the 2nd channel
	double	dNormMean2; 	// normalisation mean value for the 3rd channel
	double	dNormStd0;		// normalisation std value for the 1st channel
	double	dNormStd1;		// normalisation std value for the 2nd channel
	double	dNormStd2;		// normalisation std value for the 3rd channel

	_NetDetailsConfig(int _nDeviceID = -1, 
		double _dNM0 = 0.0f, double _dNM1 = 0.0f, double _dNM2 = 0.0f, 
		double _dNS0 = 1.0f, double _dNS1 = 1.0f, double _dNS2 = 1.0f)
	{
		nDeviceID = _nDeviceID;
		dNormMean0 = _dNM0;
		dNormMean1 = _dNM1;
		dNormMean2 = _dNM2;
		dNormStd0 = _dNS0;
		dNormStd1 = _dNS1;
		dNormStd2 = _dNS2;
	}

}NetDetailsConfig;


//########################################################################
// Object detection related data structures and types
//########################################################################

// Structure to hold the common and high-level information across all the object detection networks
// The new fields maybe added to this structure in future
typedef struct _ObjDetNetConfig
{
	float fConfThresh;		// confidence threshold
	float fNMSThresh;		// non-maximum suppression threshold
	std::string sModelPath;	// path to model weights
	std::string sClassPath;	// path to class names


	_ObjDetNetConfig(float _fCT = 0.5f, float _fNT = 0.5f, std::string _sMP = "", std::string _sCP = "")
	{
		fConfThresh = _fCT;
		fNMSThresh = _fNT;
		sModelPath = _sMP;
		sClassPath = _sCP;
	}
}ObjDetNetConfig;

// Structure to hold bounding box information of detected objects
typedef struct _ObjBBox
{
	float	fX1;		// top left x
	float	fY1;		// top left y
	float	fX2;		// bottom right x
	float	fY2;		// bottom right y
	float	fScore;		// confidence/score
	int		nClassID;	// predicted class ID. range [0, classes-1]. -1 means invalid


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

typedef std::vector<ObjBBox>		ObjBoxArr;	// array type of bounding boxes	
typedef std::vector<std::string>	ObjClsArr;	// array type of class names


//########################################################################
// ReID related data structures and types
//########################################################################

// Structure to hold the common and high-level information across all the ReID networks
typedef struct _ReIDNetConfig
{
	int		nTopK;			// top K similar images to be returned
	float	fSimThresh;		// similarity threshold between two feature vectors/images
	std::string sModelPath;	// path to model weights

	_ReIDNetConfig(int _nTK = 10, float _fST = 0.5f, std::string _sMP = "")
	{
		nTopK = _nTK;
		fSimThresh = _fST;
		sModelPath = _sMP;
	}
}ReIDNetConfig;


// Structure to hold the result of ReID
typedef struct _ReIDRes
{
	int 	nRank;			// similarity rank of the image among the gallery images. 0: top-1, 1: top-2, etc.
	int		nImgID;			// gallery image ID
	float	fSimilarity;	// similarity score between query and gallery image/features

	_ReIDRes(int _nRank = 0, int _nImgID = -1, float _fSim = 0.0f)
	{
		nRank = _nRank;
		nImgID = _nImgID;
		fSimilarity = _fSim;
	}
}ReIDRes;

typedef std::vector<ReIDRes>	ReIDResArr;		// array type of ReID results