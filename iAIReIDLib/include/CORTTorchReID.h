#pragma once
#include "type_define.h"
#include <opencv2/opencv.hpp>
#include "CReID.h"
#include "CORTInferer.h"

// Class for YouReID deep network using ORT(ONNX Runtime) inference engine
class IAIREIDLIB_API CORTTorchReID : public CReID, public CORTInferer
{
public:
	CORTTorchReID(const ReIDNetConfig& stReIDNetConfig, const NetDetailsConfig& stNetDetailsConfig);
	~CORTTorchReID();

	// Perform ReID between the query image and the gallery images
	// @param[in] cvQueryImg: single query image
	// @param[in] cvGalleryImgs: multiple gallery images
	// @return: ReID results
	virtual const ReIDResArr& ReID(const cv::Mat& cvQueryImg, const std::vector<cv::Mat>& cvGalleryImgs);

	// Perform ReID between the query embedding feature and the gallery images
	// @param[in] vQueryFeature: query embedding feature
	// @param[in] cvGalleryImgs: multiple gallery images
	// @return: ReID results
	virtual const ReIDResArr& ReID(const std::vector<float>& vQueryFeature, const std::vector<cv::Mat>& cvGalleryImgs);

	// Extract the feature vector from the input image
	// @param[in] cvImg: input image
	// @param[out] vFeature: extracted feature vector
	// @return: true if the feature is successfully extracted, false otherwise
	virtual const bool ExtractFeature(const cv::Mat& cvImg, std::vector<float>& vFeature);
protected:
	// Preprocess the input image
	// @param[in] cvImg: input image to be preprocessed
	// @param[out] cvProcImg: preprocessed image
	virtual void PreProcess(const cv::Mat& cvImg, cv::Mat& cvProcImg);

	// Postprocess the output tensor
	// @param[in] cvOrgImgSize: original image size
	// @param[in] pTensorData: output tensor data to be postprocessed
	// @param[out] pPostProcessData: postprocessed data
	// [Note] - The normalised feature vector is stored in pPostProcessData
	virtual void PostProcess(const cv::Size& cvOrgImgSize, const void* pTensorData, void* pPostProcessData);

};
