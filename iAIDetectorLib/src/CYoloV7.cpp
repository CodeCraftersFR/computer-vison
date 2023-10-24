#include "CYoloV7.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include "CORTPars.h"



CYoloV7::CYoloV7(const YoloNetConfig& stNetConfig)
	: m_nNetInputH(0)
	, m_nNetInputW(0)
	, m_nNetOutputs(0)
	, m_nNetProposals(0)
	, m_fConfThresh(0)
	, m_fNMSThresh(0)
	, m_pORTPars(nullptr)
{
	try {
		m_pORTPars = new CORTPars();

		m_fConfThresh = stNetConfig.fConfThresh;
		m_fNMSThresh = stNetConfig.fNMSThresh;


		std::string model_path = stNetConfig.sModelPath;
		std::wstring widestr = std::wstring(model_path.begin(), model_path.end());
		//OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0);

		m_pORTPars->env = Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "ONNX_YOLOV7_DETECTION");
		m_pORTPars->sessionOptions = Ort::SessionOptions();
		m_pORTPars->sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
		m_pORTPars->session = Ort::Session(m_pORTPars->env, widestr.c_str(), m_pORTPars->sessionOptions);

		size_t nNumInputNodes = m_pORTPars->session.GetInputCount();
		size_t nNnumOutputNodes = m_pORTPars->session.GetOutputCount();
		
		Ort::AllocatorWithDefaultOptions allocator;
		for (int i = 0; i < nNumInputNodes; i++)
		{
			auto inputName = m_pORTPars->session.GetInputNameAllocated(i, allocator);
			m_pORTPars->m_vInputNamesPtr.push_back(std::move(inputName));
			m_pORTPars->m_vInputNames.push_back(m_pORTPars->m_vInputNamesPtr.back().get());

			Ort::TypeInfo inputTypeInfo = m_pORTPars->session.GetInputTypeInfo(i);
			auto inputDims = inputTypeInfo.GetTensorTypeAndShapeInfo().GetShape();
			m_vNetInputNodeDims.push_back(inputDims);
		}
		for (int i = 0; i < nNnumOutputNodes; i++)
		{
			auto outputName = m_pORTPars->session.GetOutputNameAllocated(i, allocator);
			m_pORTPars->m_vOutputNamesPtr.push_back(std::move(outputName));
			m_pORTPars->m_vOutputNames.push_back(m_pORTPars->m_vOutputNamesPtr.back().get());


			Ort::TypeInfo outputTypeInfo = m_pORTPars->session.GetOutputTypeInfo(i);
			auto outputDims = outputTypeInfo.GetTensorTypeAndShapeInfo().GetShape();
			m_vNetOuputNodeDims.push_back(outputDims);
		}
		m_nNetInputH = m_vNetInputNodeDims[0][2];
		m_nNetInputW = m_vNetInputNodeDims[0][3];
		m_nNetOutputs = m_vNetOuputNodeDims[0][2];
		m_nNetProposals = m_vNetOuputNodeDims[0][1];

		std::ifstream ifs(stNetConfig.sClassPath.c_str());
		std::string line;
		while (std::getline(ifs, line)) m_vClassNames.push_back(line);
	}
	catch (Ort::Exception& e)
	{
		const char* msg = e.what();
		std::cout << msg << std::endl;
	}
	catch (std::exception& e)
	{
		const char* msg = e.what();
		std::cout << msg << std::endl;
	}
}

CYoloV7::CYoloV7()
	: m_nNetInputH(0)
	, m_nNetInputW(0)
	, m_nNetOutputs(0)
	, m_nNetProposals(0)
	, m_fConfThresh(0)
	, m_fNMSThresh(0)
	, m_pORTPars(nullptr)
{

}

CYoloV7::~CYoloV7()
{
	if(m_pORTPars)
		delete m_pORTPars;	m_pORTPars = nullptr;
}

// Preprocess the image to feed into the network
// @param cvImg: input image
// @return blobImage: preprocessed image
cv::Mat CYoloV7::PreProcess(cv::Mat& cvImg)
{
	// Channels order: BGR to RGB and resize
	cv::Mat cvResizedImg;
	cv::cvtColor(cvImg, cvResizedImg, cv::COLOR_BGR2RGB);
	cv::resize(cvResizedImg, cvResizedImg, cv::Size(m_nNetInputW, m_nNetInputH));

	// Convert image to float32 and normalize
	cv::Mat floatImage;
	cvResizedImg.convertTo(floatImage, CV_32F, 1.0 / 255.0);

	// Create a 4-dimensional blob from the image
	cv::Mat blobImage = cv::dnn::blobFromImage(floatImage);

	return blobImage;
}

// Postprocess the tensor output of the network
// @param cvOrgImgSize: original image size
// @param pOutputTensorData: tensor data returned by the network
// @return vDetObj: detected objects
std::vector<ObjBBox> CYoloV7::PostProcess(cv::Size cvOrgImgSize, void* pOutputTensorData)
{
	std::vector<Ort::Value>& outputTensors = *(std::vector<Ort::Value>*)pOutputTensorData;

	Ort::Value& predictions = outputTensors.at(0);
	

	float ratioh = (float)cvOrgImgSize.height / m_nNetInputH, ratiow = (float)cvOrgImgSize.width / m_nNetInputW;
	const float* pData = predictions.GetTensorMutableData<float>();

	std::vector<ObjBBox> vDetObj;
	for (int n = 0, nClassNum = m_vClassNames.size(); n < m_nNetProposals; n++)   
	{
		float fScore = pData[4];
		if (fScore > m_fConfThresh)
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
			if (fMaxClassScore > m_fConfThresh)
			{
				float cx = pData[0] * ratiow;  ///cx
				float cy = pData[1] * ratioh;   ///cy
				float w = pData[2] * ratiow;   ///w
				float h = pData[3] * ratioh;  ///h

				float xmin = cx - 0.5 * w;
				float ymin = cy - 0.5 * h;
				float xmax = cx + 0.5 * w;
				float ymax = cy + 0.5 * h;

				vDetObj.push_back(ObjBBox{ xmin, ymin, xmax, ymax, fMaxClassScore, nMaxClassID });
			}
		}
		pData += m_nNetOutputs;
	}


	NMSBoxes(vDetObj);

	return vDetObj;
}

// Perform non-maximum suppression
// @param vDetObj: detected objects before NMS
// @return vDetObj: detected objects after NMS
void CYoloV7::NMSBoxes(std::vector<ObjBBox>& vDetObj)
{
	sort(vDetObj.begin(), vDetObj.end(), [](ObjBBox a, ObjBBox b) { return a.fScore > b.fScore; });
	std::vector<float> vArea(vDetObj.size());
	for (int i = 0; i < int(vDetObj.size()); ++i)
	{
		vArea[i] = (vDetObj.at(i).fX2 - vDetObj.at(i).fX1 + 1)
			* (vDetObj.at(i).fY2 - vDetObj.at(i).fY1 + 1);
	}

	std::vector<bool> vIsSupressed(vDetObj.size(), false);
	for (int i = 0; i < int(vDetObj.size()); ++i)
	{
		if (vIsSupressed[i]) { continue; }
		for (int j = i + 1; j < int(vDetObj.size()); ++j)
		{
			if (vIsSupressed[j]) { continue; }
			float fX1 = _MAX(vDetObj[i].fX1, vDetObj[j].fX1);
			float fY1 = _MAX(vDetObj[i].fY1, vDetObj[j].fY1);
			float fX2 = _MIN(vDetObj[i].fX2, vDetObj[j].fX2);
			float fY2 = _MIN(vDetObj[i].fY2, vDetObj[j].fY2);

			float fW = _MAX(float(0), fX2 - fX1 + 1);
			float fH = _MAX(float(0), fY2 - fY1 + 1);
			float fInterArea = fW * fH;
			float fOverlap = fInterArea / (vArea[i] + vArea[j] - fInterArea);

			if (fOverlap >= m_fNMSThresh)
			{
				vIsSupressed[j] = true;
			}
		}
	}
	// return post_nms;
	int idx_t = 0;
	vDetObj.erase(remove_if(vDetObj.begin(), vDetObj.end(), [&idx_t, &vIsSupressed](const ObjBBox& f) { return vIsSupressed[idx_t++]; }), vDetObj.end());
}

// Detects objects in the input frame
// @param cvFrame: input frame
void CYoloV7::Detect(cv::Mat& cvFrame)
{
	try
	{
		cv::Mat inputImg = PreProcess(cvFrame);

		
		std::array<int64_t, 4> inputDims{ 1, 3, m_nNetInputH, m_nNetInputW };

		Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu( OrtAllocatorType::OrtDeviceAllocator, 
			                                                     OrtMemType::OrtMemTypeDefault);

		Ort::Value inputTensor = Ort::Value::CreateTensor<float>( memoryInfo, 
																  inputImg.ptr<float>(),
																  inputImg.total() * sizeof(float),
			                                                      inputDims.data(), 
			                                                      inputDims.size());


		std::vector<Ort::Value> outputTensors = m_pORTPars->session.Run(Ort::RunOptions{ nullptr }, 
			                                                            &m_pORTPars->m_vInputNames[0], 
			                                                            &inputTensor, 
																		m_vNetInputNodeDims.size(), 
			                                                            m_pORTPars->m_vOutputNames.data(), 
			                                                            m_pORTPars->m_vOutputNames.size());

		
		std::vector<ObjBBox> vDetObj = PostProcess(cvFrame.size(), &outputTensors);


		for (size_t i = 0; i < vDetObj.size(); ++i)
		{
			int xmin = int(vDetObj[i].fX1);
			int ymin = int(vDetObj[i].fY1);
			rectangle(cvFrame, cv::Point(xmin, ymin), cv::Point(int(vDetObj[i].fX2), int(vDetObj[i].fY2)), cv::Scalar(0, 0, 255), 2);
			std::string label = cv::format("%.2f", vDetObj[i].fScore);
			label = m_vClassNames[vDetObj[i].nClassID] + ":" + label;
			putText(cvFrame, label, cv::Point(xmin, ymin - 5), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 255, 0), 1);
		}
	}
	catch (Ort::Exception& e)
	{
		const char* msg = e.what();
		std::cout << msg << std::endl;
	}
	catch (cv::Exception& e)
	{
		const char* msg = e.what();
		std::cout << msg << std::endl;
	}
	catch (std::exception& e)
	{
		const char* msg = e.what();
		std::cout << msg << std::endl;
	}
}