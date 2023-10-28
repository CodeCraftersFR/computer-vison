#include "CORTInferer.h"
#include "CORTPars.h"


CORTInferer::CORTInferer(const NetDetailsConfig& stConfig)
	: m_NetDetailsConfig(stConfig)
	, m_pORTPars(nullptr)
	, m_nNetInputW(0)
	, m_nNetInputH(0)
	, m_nNetOutputs(0)
	, m_nNetProposals(0)
	, m_bValid(false)
{

}

CORTInferer::~CORTInferer()
{
	if (!m_pORTPars)
	{
		delete m_pORTPars;	m_pORTPars = nullptr;
	}
}

// Main function to run inference
// @param[in] cvFrame: input image
// @param[out] pResultData: output data
// @return: true if success, otherwise false
bool CORTInferer::Infer(const cv::Mat& cvFrame, void* pResultData)
{
	if (!m_bValid)
		return false;

	if (cvFrame.empty())
		return false;

	if (pResultData == nullptr)
		return false;

	try
	{
		cv::Mat cvInputImg;
		PreProcess(cvFrame, cvInputImg);


		std::array<int64_t, 4> inputDims{ 1, 3, m_nNetInputH, m_nNetInputW };

		Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtDeviceAllocator,
			OrtMemType::OrtMemTypeDefault);

		Ort::Value inputTensor = Ort::Value::CreateTensor<float>(memoryInfo,
			cvInputImg.ptr<float>(),
			cvInputImg.total() * sizeof(float),
			inputDims.data(),
			inputDims.size());


		std::vector<Ort::Value> outputTensors = m_pORTPars->session.Run(Ort::RunOptions{ nullptr },
			&m_pORTPars->m_vInputNames[0],
			&inputTensor,
			m_vNetInputNodeDims.size(),
			m_pORTPars->m_vOutputNames.data(),
			m_pORTPars->m_vOutputNames.size());


		PostProcess(cvFrame.size(), &outputTensors, pResultData);


	}
	catch (Ort::Exception& e)
	{
		const char* msg = e.what();
		std::cout << msg << std::endl;
		return false;
	}
	catch (cv::Exception& e)
	{
		const char* msg = e.what();
		std::cout << msg << std::endl;
		return false;
	}
	catch (std::exception& e)
	{
		const char* msg = e.what();
		std::cout << msg << std::endl;
		return false;
	}

	return true;
}

// TODO: Implement the GPU device configuration in the future. Currently, only CPU is supported as requested by the client.
// Read the onnx deep learning model
// @param[in] sModelPath: path to the onnx model
// @param[in] sPairedFilePath: path to the file paired with the onnx model. e.g., the config file or the class name file
// @param[in] sLogID: log id of onnxruntime
// @return: true if success, otherwise false
bool CORTInferer::ReadModel(const std::string& sModelPath, const std::string& sPairedFilePath/* = ""*/, const std::string& sLogID /*= "onnxruntime"*/)
{
	(void)sPairedFilePath;

	try {
		if(!m_pORTPars)
			m_pORTPars = new CORTPars();


		std::wstring widestr = std::wstring(sModelPath.begin(), sModelPath.end());

		// Set log level and log id of onnxruntime
		m_pORTPars->env = Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, sLogID.c_str());

		// Initialise and set session options
		m_pORTPars->sessionOptions = Ort::SessionOptions();
		m_pORTPars->sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);

		// Initialise session with model
		m_pORTPars->session = Ort::Session(m_pORTPars->env, widestr.c_str(), m_pORTPars->sessionOptions);

		Ort::AllocatorWithDefaultOptions allocator;
		
		// Get input node dims and names
		size_t nNumInputNodes = m_pORTPars->session.GetInputCount();
		for (int i = 0; i < nNumInputNodes; i++)
		{
			auto inputName = m_pORTPars->session.GetInputNameAllocated(i, allocator);
			m_pORTPars->m_vInputNamesPtr.push_back(std::move(inputName));
			m_pORTPars->m_vInputNames.push_back(m_pORTPars->m_vInputNamesPtr.back().get());

			Ort::TypeInfo inputTypeInfo = m_pORTPars->session.GetInputTypeInfo(i);
			auto inputDims = inputTypeInfo.GetTensorTypeAndShapeInfo().GetShape();
			m_vNetInputNodeDims.push_back(inputDims);
		}

		// Get output node dims and names
		size_t nNnumOutputNodes = m_pORTPars->session.GetOutputCount();
		for (int i = 0; i < nNnumOutputNodes; i++)
		{
			auto outputName = m_pORTPars->session.GetOutputNameAllocated(i, allocator);
			m_pORTPars->m_vOutputNamesPtr.push_back(std::move(outputName));
			m_pORTPars->m_vOutputNames.push_back(m_pORTPars->m_vOutputNamesPtr.back().get());


			Ort::TypeInfo outputTypeInfo = m_pORTPars->session.GetOutputTypeInfo(i);
			auto outputDims = outputTypeInfo.GetTensorTypeAndShapeInfo().GetShape();
			m_vNetOuputNodeDims.push_back(outputDims);
		}

		// Get network basic info such as input size, output size, etc
		m_nNetInputH = (int)m_vNetInputNodeDims[0][2];
		m_nNetInputW = (int)m_vNetInputNodeDims[0][3];
		m_nNetOutputs = (int)m_vNetOuputNodeDims[0][2];
		m_nNetProposals = (int)m_vNetOuputNodeDims[0][1];
	}
	catch (Ort::Exception& e)
	{
		const char* msg = e.what();
		std::cout << msg << std::endl;
		return false;
	}
	catch (std::exception& e)
	{
		const char* msg = e.what();
		std::cout << msg << std::endl;
		return false;
	}

	return true;
}

// Core function for preprocessing the input image
// @param[in] cvImg: input image to be preprocessed
// @param[in] bSwapRB: whether to swap the R and B channels
// @param[out] cvProcImg: preprocessed image
void CORTInferer::PreProcessCore(const cv::Mat& cvImg, bool bSwapRB, cv::Mat& cvProcImg) const
{
	// If all the std values are same, use the blobFromImage function directly
	if(m_NetDetailsConfig.dNormStd0 == m_NetDetailsConfig.dNormStd1 && 
		m_NetDetailsConfig.dNormStd0 == m_NetDetailsConfig.dNormStd2)
	{
		cvProcImg = cv::dnn::blobFromImage(
			cvImg,
			m_NetDetailsConfig.dNormStd0,
			cv::Size(m_nNetInputW, m_nNetInputH),
			cv::Scalar(m_NetDetailsConfig.dNormMean0, m_NetDetailsConfig.dNormMean1, m_NetDetailsConfig.dNormMean2),
			bSwapRB);
	}
	else // Otherwise
	{
		cv::Mat cvTmp = cv::Mat(cvImg.rows, cvImg.cols, CV_32FC3);
		for (int y = 0; y < cvTmp.rows; y++)
		{
			for (int x = 0; x < cvTmp.cols; x++)
			{
				cvTmp.at<cv::Vec3f>(y, x)[0] = (float)(cvImg.at<cv::Vec3b>(y, x)[0] - m_NetDetailsConfig.dNormMean0) * m_NetDetailsConfig.dNormStd0;
				cvTmp.at<cv::Vec3f>(y, x)[1] = (float)(cvImg.at<cv::Vec3b>(y, x)[1] - m_NetDetailsConfig.dNormMean1) * m_NetDetailsConfig.dNormStd1;
				cvTmp.at<cv::Vec3f>(y, x)[2] = (float)(cvImg.at<cv::Vec3b>(y, x)[2] - m_NetDetailsConfig.dNormMean2) * m_NetDetailsConfig.dNormStd2;
			}
		}

		cvProcImg = cv::dnn::blobFromImage(
			cvTmp,
			1.0,
			cv::Size(m_nNetInputW, m_nNetInputH),
			cv::Scalar(0, 0, 0),
			bSwapRB);
	}

	
}

const void* CORTInferer::PostProcessCore(const void* pTensorData) const
{
	std::vector<Ort::Value>& outputTensors = *(std::vector<Ort::Value>*)pTensorData;

	Ort::Value& predictions = outputTensors.at(0);

	const float* pData = predictions.GetTensorMutableData<float>();

	return (const void*)pData;
}

// Validate the required parameters
bool CORTInferer::Validate()
{
	if (!m_pORTPars)
		return false;


	if (m_nNetInputW == 0 || m_nNetInputH == 0)
		return false;

	if (m_nNetOutputs == 0)
		return false;

	if (m_nNetProposals == 0)
		return false;

	return true;
}

