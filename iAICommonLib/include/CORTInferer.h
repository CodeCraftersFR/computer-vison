#pragma once
#include "type_define.h"
#include <opencv2/opencv.hpp>


class CORTPars;

// Base abstract class for onnxruntime inference.
// All the specific onnxruntime inference classes should inherit from this class.
class IAICOMMONLIB_API CORTInferer
{
public:
	CORTInferer(const NetDetailsConfig& stConfig);

	virtual ~CORTInferer();

	// Check whether the network is valid or not
	// @return: true if valid, otherwise false
	bool IsValid() const { return m_bValid; }

	// Do inference on the input image using onnxruntime
	// @param[in] cvFrame: input image
	// @param[out] pResultData: output data
	// @return: true if success, otherwise false
	virtual bool Infer(const cv::Mat& cvFrame, void* pResultData);

	// Read the onnx model from the given path
	// @param[in] sModelPath: path to the onnx model
	// @param[in] sPairedFilePath: path to the paired file
	// @param[in] sLogTitle: title for logging
	// @return: true if success, otherwise false
	virtual bool ReadModel(const std::string& sModelPath, const std::string& sPairedFilePath = "", const std::string& sLogTitle = "onnxruntime");

protected:
	// Preprocess the input image
	// @param[in] cvImg: input image to be preprocessed
	// @param[out] cvProcImg: preprocessed image
	virtual void PreProcess(const cv::Mat& cvImg, cv::Mat& cvProcImg) = 0;

	// Postprocess the output tensor
	// @param[in] cvOrgImgSize: original image size
	// @param[in] pTensorData: output tensor data to be postprocessed
	// @param[out] pPostProcessData: postprocessed data
	virtual void PostProcess(const cv::Size& cvOrgImgSize, const void* pTensorData, void* pPostProcessData) = 0;

	// Validate the network configuration before inference
	virtual bool Validate();


protected:
	// Core function for preprocessing the input image
	// @param[in] cvImg: input image to be preprocessed
	// @param[in] bSwapRB: whether to swap the R and B channels
	// @param[out] cvProcImg: preprocessed image
	void PreProcessCore(const cv::Mat& cvImg, bool bSwapRB, cv::Mat& cvProcImg) const;
	
	// Core function for postprocessing the input image
	// @param[in] pTensorData: output tensor data to be postprocessed
	// @return: postprocessed data pointer by a default core function
	const void* PostProcessCore(const void* pTensorData) const;

protected:
	NetDetailsConfig m_NetDetailsConfig;	// network details configuration
	bool	m_bValid;						// whether the network is valid or not

	int		m_nNetInputW;					// input image width to the network
	int		m_nNetInputH;					// input image height to the network
	int		m_nNetOutputs;					// number of network outputs
	int		m_nNetProposals;				// number of proposals of network

private:
	std::vector<std::vector<int64_t>> m_vNetInputNodeDims;
	std::vector<std::vector<int64_t>> m_vNetOuputNodeDims;

	CORTPars* m_pORTPars;					// ONNX Runtime parameters

};