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
	// @param[in] cvFrame: input frame in BGR format
	// @return: true if detection is successful, false otherwise. 
	// [Note]: The bounding boxes of detected objects can be obtained by calling GetObjBoxes().
	virtual bool Detect(const cv::Mat& cvFrame);

	// Read the onnx model from the given path
	// @param[in] sModelPath: path to the onnx model
	// @param[in] sPairedFilePath: path to the paired file. For this case, it is the path to the class names file
	// @param[in] sLogTitle: title of the log
	// @return: true if successfully read the model, false otherwise
	virtual bool ReadModel(const std::string& sModelPath, const std::string& sPairedFilePath = "", const std::string& sLogTitle = "onnxruntime");


protected:
	// Preprocess the input image
	// @param[in] cvImg: input image to be preprocessed
	// @param[out] cvProcImg: preprocessed image
	virtual void PreProcess(const cv::Mat& cvImg, cv::Mat& cvProcImg);

	// Postprocess the output tensor
	// @param[in] cvOrgImgSize: size of the original image
	// @param[in] pTensorData: pointer to the output tensor data of the inference engine
	// @param[out] pPostProcessData: pointer to the postprocessed data
	virtual void PostProcess(const cv::Size& cvOrgImgSize, const void* pTensorData, void* pPostProcessData);
};