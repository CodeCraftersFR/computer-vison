#pragma once
#include "type_define.h"
#include <opencv2/opencv.hpp>
#include "CORTInferer.h"
#include "CObjDetector.h"

// Class for YOLOv7 object detection using ORT(ONNX Runtime) inference engine
class IAIDETECTORLIB_API CORTYoloV7 : public CObjDetector, public CORTInferer
{
public:
	
	CORTYoloV7(const ObjDetNetConfig& stObjDetNetConfig, const NetDetailsConfig& stNetDetailsConfig);
	~CORTYoloV7();

	// Detect objects in the input frame
	virtual const ObjBoxArr& Detect(cv::Mat& cvFrame);

	// Read the onnx model from the given path
	virtual bool ReadModel(const std::string& sModelPath, const std::string& sPairedFilePath = "", const std::string& sLogTitle = "onnxruntime");


protected:
	// Preprocess the input image
	// @param[in] cvImg: input image to be preprocessed
	// @param[out] cvProcImg: preprocessed image
	virtual void PreProcess(const cv::Mat& cvImg, cv::Mat& cvProcImg);

	// Postprocess the output tensor
	virtual void PostProcess(const cv::Size& cvOrgImgSize, const void* pTensorData, void* pPostProcessData);
};