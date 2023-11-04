#include "CAIAnalysis.h"
#include "macro_define.h"
#include "CORTYoloV7.h"
#include "CORTYouReID.h"
#include "CORTTorchReID.h"
#include "CVideoWriter.h"

#define DEVICE_ID	-1

CAIAnalysis::CAIAnalysis(const S_AnalysisParam& stParam)
	: m_stParam(stParam)
	, m_pObjDetector(nullptr)
	, m_pReID(nullptr)
	, m_pVideoWriter(nullptr)
	, m_eWriteResultType(E_AnalysisTaskType::eAttUnknown)
	, m_bValid(false)
{
	m_bValid = Init();
}

CAIAnalysis::~CAIAnalysis()
{
	Release();
}

// Run the given analysis task
// @param[in] eTaskType: the type of analysis task
// @param[in] cvBGRFrame: the input BGR format frame
// @return true if the task is run successfully, otherwise false
bool CAIAnalysis::RunTask(const E_AnalysisTaskType& eTaskType, const cv::Mat& cvBGRFrame)
{
	bool bRes = false;

	if (!m_bValid)
		return false;

	if (eTaskType == E_AnalysisTaskType::eAttPersonDetection)
	{
		bRes = RunDetection(cvBGRFrame);
	}
	else if (eTaskType == E_AnalysisTaskType::eAttPersonRegister)
	{
		bRes = RunRegistration(cvBGRFrame);
	}
	else if (eTaskType == E_AnalysisTaskType::eAttPersonReID)
	{
		bRes = RunReID(cvBGRFrame);
	}
	else
	{
		return false;
	}

	if (!bRes)
		return false;

	bRes = WriteResultVideo(cvBGRFrame);

	return bRes;
}

// Begin to write task results to video, given the result type
// @param[in] eWriteTaskType: the type of result to write to video
// @param[in] sVideoPath: the path of the video to write. The file extension should be "mp4" or "avi".
// @param[in] nFPS: the FPS of the video to write
// @param[in] nW: the width of the video to write
// @param[in] nH: the height of the video to write
// @return true if the video writer is opened successfully, otherwise false
// [Note] - The relevant video codec will be determined by the file extension automatically.
//        - The task result will be written to video from the point of calling this function.
bool CAIAnalysis::BeginVideoWriter(const E_AnalysisTaskType& eWriteTaskType, const std::string& sVideoPath, int nFPS, int nW, int nH)
{
	if (!m_pVideoWriter)
		return false;

	m_eWriteResultType = eWriteTaskType;

	if(!m_pVideoWriter->Open(sVideoPath, nW, nH, nFPS) || !m_pVideoWriter->IsValid())
	{
		return false;
	}

	return true;
}

// End to write task results to video
// @return true if the video writer is closed successfully, otherwise false
bool CAIAnalysis::EndVideoWriter()
{
	m_eWriteResultType = E_AnalysisTaskType::eAttUnknown;

	if(!m_pVideoWriter || !m_pVideoWriter->IsValid())
		return true;

	m_pVideoWriter->Release();

	return true;
}

// Get the detection result
// @return the detection result
const ObjBoxArr* CAIAnalysis::GetDetectionResult() const
{
	if (!m_bValid)
		return nullptr;

	return &m_pObjDetector->GetObjBoxes();
}

// Get the re-identification result
// @return the re-identification result
const ReIDResArr* CAIAnalysis::GetReIDResult() const
{
	if (!m_bValid)
		return nullptr;

	return &m_pReID->GetReIDRes();
}

// Check if the analysis library is valid
// @return true if the analysis library is valid, otherwise false
const bool CAIAnalysis::IsValid() const
{
	return m_bValid;
}

// Draw the result on the given frame for the given task type.
// @param[in] eDrawTaskType: the type of analysis task
// @param[in] bCheckResultExistence: true if the result existence should be checked, otherwise false
//            If the result existence is checked, the result will be drawn only if the result exists. otherwise return false.
// @param[in/out] cvBGRFrame: the input BGR format frame
bool CAIAnalysis::DrawResult(const E_AnalysisTaskType& eDrawTaskType, bool bCheckResultExistence, cv::Mat& cvBGRFrame)
{
	if (!m_bValid)
		return false;

	if (eDrawTaskType == E_AnalysisTaskType::eAttPersonDetection)
	{
		// If the result existence should be checked, the result will be drawn only if the result exists.
		if (bCheckResultExistence && m_pObjDetector->GetObjBoxes().empty())
			return false;

		m_pObjDetector->DrawBBox(cvBGRFrame, m_pObjDetector->GetObjBoxes(), false, true);
	}
	else if (eDrawTaskType == E_AnalysisTaskType::eAttPersonReID)
	{
		const ObjBoxArr& vObjBoxes = m_pObjDetector->GetObjBoxes();
		const ReIDResArr& vReIDRes = m_pReID->GetReIDRes();

		// If the result existence should be checked, the result will be drawn only if the result exists.
		if(bCheckResultExistence && (vObjBoxes.empty() || vReIDRes.empty()))
			return false;

		ObjBoxArr vDrawBoxes;
		for (int i = 0; i < vReIDRes.size(); i++)
		{
			int nImgId = vReIDRes[i].nImgID;
			if (nImgId < 0 || nImgId >= vObjBoxes.size())
				continue;

			ObjBBox stObjBox = vObjBoxes[nImgId];
			stObjBox.fScore = vReIDRes[i].fSimilarity;
			vDrawBoxes.push_back(stObjBox);
		}

		m_pObjDetector->DrawBBox(cvBGRFrame, vDrawBoxes, false, true);
	}
	else 
	{
		return false;
	}

	return true;
}

bool CAIAnalysis::Init()
{
	if (!InitObjDetector())
		return false;

	if (!InitReID())
		return false;

	if (!InitVideoWriter())
		return false;

	return true;
}

bool CAIAnalysis::InitObjDetector()
{
	if (m_stParam.eDeviceType != E_DeviceType::eDtCPU)
		return false;

	if (m_stParam.eRuntimeType != E_InferenceRuntimeType::eIrtOnnx)
		return false;

	if (m_stParam.eDetectionMode != E_DetectionMode::eDMYoloV7)
		return false;

	ObjDetNetConfig stObjDetNetConfig = {
		m_stParam.fDetConfThresh,
		0.5f,
		DL_YOLO7_OBJ_DET_ONNX_MODEL_PATH,
		DL_YOLO7_OBJ_CLASS_NAME_PATH
	};

	NetDetailsConfig stNetDetailsConfig = {
		DEVICE_ID,
		0, 0, 0,
		(double)1.0 / 255,
		(double)1.0 / 255,
		(double)1.0 / 255
	};


	m_pObjDetector = new CORTYoloV7(stObjDetNetConfig, stNetDetailsConfig);
	if (!m_pObjDetector)
		return false;

	if (!((CORTYoloV7*)m_pObjDetector)->IsValid())
	{
		delete m_pObjDetector; m_pObjDetector = nullptr;
		return false;
	}

	// Limit object detector to detect person only
	ObjClsArr vClsNames2Detect{ "person" };
	m_pObjDetector->SetClsNames2Detect(vClsNames2Detect);

	return true;
}

bool CAIAnalysis::InitReID()
{
	if (m_stParam.eDeviceType != E_DeviceType::eDtCPU)
		return false;

	if (m_stParam.eRuntimeType != E_InferenceRuntimeType::eIrtOnnx)
		return false;

	if (m_stParam.eReIDMode == E_ReIDMode::eRmTorchReID)
	{
		ReIDNetConfig stReIDNetCfg = {
			m_stParam.nReIDTopK,
			m_stParam.fReIDConfThresh,
			DL_TORCHREID_ONNX_MODEL_PATH,
		};

		NetDetailsConfig stTorchReIDNetDetailsCfg = {
			DEVICE_ID,
			(double)0.406f * 255,
			(double)0.456f * 255,
			(double)0.485f * 255,
			(double)1.0 / ((double)0.225f * 255),
			(double)1.0 / ((double)0.224f * 255),
			(double)1.0 / ((double)0.229f * 255),
		};

		m_pReID = new CORTTorchReID(stReIDNetCfg, stTorchReIDNetDetailsCfg);
		if (!m_pReID)
			return false;

		if (!((CORTTorchReID*)m_pReID)->IsValid())
		{
			delete m_pReID; m_pReID = nullptr;
			return false;
		}
	}
	else if (m_stParam.eReIDMode == E_ReIDMode::eRmYouReID)
	{
		ReIDNetConfig stReIDNetCfg = {
			m_stParam.nReIDTopK,
			m_stParam.fReIDConfThresh,
			DL_YOUREID_ONNX_MODEL_PATH,
		};

		NetDetailsConfig stYouReIDNetDetailsCfg = {
			DEVICE_ID,
			(double)0.406f * 255,
			(double)0.456f * 255,
			(double)0.485f * 255,
			(double)1.0 / ((double)0.225f * 255),
			(double)1.0 / ((double)0.224f * 255),
			(double)1.0 / ((double)0.229f * 255),
		};

		m_pReID = new CORTYouReID(stReIDNetCfg, stYouReIDNetDetailsCfg);
		if (!m_pReID)
			return false;

		if (!((CORTYouReID*)m_pReID)->IsValid())
		{
			delete m_pReID; m_pReID = nullptr;
			return false;
		}
	}
	else
	{
		return false;
	}

	return true;

}

bool CAIAnalysis::InitVideoWriter()
{
	// Create video writer backed by OpenCV Video Writer
	m_pVideoWriter = new CVideoWriter(E_WriterEngineType::eWETOpenCV);
	if (!m_pVideoWriter)
		return false;

	return true;
}

void CAIAnalysis::Release()
{
	if (m_pObjDetector)
		delete m_pObjDetector; m_pObjDetector = nullptr;

	if (m_pReID)
		delete m_pReID; m_pReID = nullptr;

	if (m_pVideoWriter)
		delete m_pVideoWriter; m_pVideoWriter = nullptr;

	m_eWriteResultType = E_AnalysisTaskType::eAttUnknown;
	m_bValid = false;
}


// Run the detection task
// @param[in] cvBGRFrame: the input BGR format frame
// @return true if the task is run successfully, otherwise false
bool CAIAnalysis::RunDetection(const cv::Mat& cvBGRFrame)
{
	if (!m_pObjDetector)
		return false;

	if (!m_pObjDetector->Detect(cvBGRFrame))
		return false;

	return true;
}

// Run the re-identification task
// @param[in] cvBGRFrame: the input BGR format frame
// @return true if the task is run successfully, otherwise false
bool CAIAnalysis::RunReID(const cv::Mat& cvBGRFrame)
{
	if (!m_pReID)
		return false;

	// Run detection first
	if (!RunDetection(cvBGRFrame))
		return false;

	// Get the detection result
	const ObjBoxArr* pDetRes = GetDetectionResult();

	// Create gallery images from the detection result by cropping the detected person
	std::vector<cv::Mat> vGalleryImgs;
	for (const ObjBBox& stObjBox : *pDetRes)
	{
		const cv::Mat& cvCropImg = cvBGRFrame(
			cv::Range((int)stObjBox.fY1, (int)stObjBox.fY2),
			cv::Range((int)stObjBox.fX1, (int)stObjBox.fX2));

		vGalleryImgs.push_back(cvCropImg);
	}

	// Run re-identification
	if (!m_pReID->ReID(vGalleryImgs))
		return false;

	return true;
}

// Run the registration task
// @param[in] cvBGRFrame: the input BGR format frame
// @return true if the task is run successfully, otherwise false
bool CAIAnalysis::RunRegistration(const cv::Mat& cvBGRFrame)
{
	if (!m_pReID)
		return false;

	if (!m_pReID->RegisterQuery(cvBGRFrame))
		return false;

	return true;
}

bool CAIAnalysis::WriteResultVideo(const cv::Mat& cvBGRFrame)
{
	if(!m_bValid)
		return false;

	if (m_eWriteResultType <= E_AnalysisTaskType::eAttUnknown || m_eWriteResultType >= E_AnalysisTaskType::eAttCount)
		return true; // must return true to ignore the case not to write result to video

	// Clone the input frame
	cv::Mat cvWriteFrame = cvBGRFrame.clone();
	

	if (!DrawResult(m_eWriteResultType, true, cvWriteFrame))
		return true;  // must return true to ignore the frame without result

	if (!m_pVideoWriter->WriteFrame(cvWriteFrame))
		return false;

	return true;
}

