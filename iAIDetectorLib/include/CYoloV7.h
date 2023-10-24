#pragma once
#include "type_define.h"
#include <opencv2/opencv.hpp>


class CORTPars;
class IAIDETECTORLIB_API CYoloV7
{
public:
	CYoloV7();
	CYoloV7(const YoloNetConfig& stNetConfig);
	~CYoloV7();

	// Detect objects in the input frame
	void Detect(cv::Mat& frame);

private:
	// Preprocess the input image
	cv::Mat PreProcess(cv::Mat& cvImg);

	// Postprocess the output tensor
	std::vector<ObjBBox> PostProcess(cv::Size originalImageSize, void* pOutputTensorData);

	// Perform non-maximum suppression
	void NMSBoxes(std::vector<ObjBBox>& vBoxes);

private:
	float	m_fConfThresh;	// confidence threshold
	float	m_fNMSThresh;     // non-maximum suppression threshold

	int		m_nNetInputW;		// input image width to the network
	int		m_nNetInputH;       // input image height to the network
	int		m_nNetOutputs;      // number of network outputs
	int		m_nNetProposals;    // number of proposals of network

	std::vector<std::string> m_vClassNames; // class names of objects which can be detected by the network
	std::vector<std::vector<int64_t>> m_vNetInputNodeDims; 
	std::vector<std::vector<int64_t>> m_vNetOuputNodeDims; 

	CORTPars* m_pORTPars;   // ONNX Runtime parameters

};