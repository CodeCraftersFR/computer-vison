#include "CORTYoloV7.h"
#include <fstream>
#include <sstream>
#include <iostream>




CORTYoloV7::CORTYoloV7(const ObjDetNetConfig& stObjDetNetConfig, const NetDetailsConfig& stNetDetailsConfig)
	: CObjDetector(stObjDetNetConfig)
	, CORTInferer(stNetDetailsConfig)
{
	m_bValid = (ReadModel(stObjDetNetConfig.sModelPath, 
		stObjDetNetConfig.sClassPath, "yolo7-onnx") && Validate());
}


CORTYoloV7::~CORTYoloV7()
{

}

// Preprocess the input image
// @param[in] cvImg: input image to be preprocessed
// @param[out] cvProcImg: preprocessed image
void CORTYoloV7::PreProcess(const cv::Mat& cvImg, cv::Mat& cvProcImg)
{
	CORTInferer::PreProcessCore(cvImg, true, cvProcImg);
}

// Postprocess the tensor output of the network
// @param cvOrgImgSize: original image size
// @param pOutputTensorData: tensor data returned by the network
// @return vDetObj: detected objects
void CORTYoloV7::PostProcess(const cv::Size& cvOrgImgSize, const void* pTensorData, void* pPostProcessData)
{
	if (!pPostProcessData)
		return;

	float fRH = (float)cvOrgImgSize.height / m_nNetInputH;
	float fRW = (float)cvOrgImgSize.width / m_nNetInputW;

	const float* pData = (const float*)CORTInferer::PostProcessCore(pTensorData);


	ObjBoxArr* pvDetObj = (ObjBoxArr*)pPostProcessData;

	for (int n = 0, nClassNum = m_vClsNames.size(); n < m_nNetProposals; n++)
	{
		float fScore = pData[4];
		if (fScore > m_stObjDetConfig.fConfThresh)
		{
			int nMaxClassID = 0;
			float fMaxClassScore = 0;
			for (int k = 0; k < nClassNum; k++)
			{
				if (pData[k + 5] > fMaxClassScore)
				{
					fMaxClassScore = pData[k + 5];
					nMaxClassID = k;
				}
			}
			fMaxClassScore *= fScore;
			if (fMaxClassScore > m_stObjDetConfig.fConfThresh)
			{
				float cx = pData[0] * fRW;  ///cx
				float cy = pData[1] * fRH;   ///cy
				float w = pData[2] * fRW;   ///w
				float h = pData[3] * fRH;  ///h

				float xmin = cx - 0.5 * w;
				float ymin = cy - 0.5 * h;
				float xmax = cx + 0.5 * w;
				float ymax = cy + 0.5 * h;

				pvDetObj->push_back(ObjBBox{ xmin, ymin, xmax, ymax, fMaxClassScore, nMaxClassID });
			}
		}
		pData += m_nNetOutputs;
	}


	NMSBoxes(pvDetObj);
}


// Detects objects in the input frame
// @param cvFrame: input frame
const ObjBoxArr& CORTYoloV7::Detect(cv::Mat& cvFrame)
{
	m_vObjBoxes.clear();

	if (!CORTInferer::Inference(cvFrame, (void*)&m_vObjBoxes))
	{
		m_vObjBoxes.clear();
		return m_vObjBoxes;
	}

	return m_vObjBoxes;
}

bool CORTYoloV7::ReadModel(const std::string& sModelPath, const std::string& sPairedFilePath /*= ""*/, const std::string& sLogTitle /*= "onnxruntime"*/)
{
	if (!CORTInferer::ReadModel(sModelPath, sPairedFilePath, sLogTitle))
		return false;
	
	m_vClsNames.clear();
	try
	{
		std::ifstream ifs(sPairedFilePath.c_str());
		std::string line;
		while (std::getline(ifs, line)) m_vClsNames.push_back(line);
	}
	catch (std::exception& e)
	{
		const char* msg = e.what();
		std::cout << msg << std::endl;
		return false;
	}
	catch (...)
	{
		std::cout << "Unknown exception occurred when reading class names." << std::endl;
		return false;
	}

	return true;
}
