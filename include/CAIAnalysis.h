#pragma once
#include <analysis_type.h>
#include <opencv2/opencv.hpp>



class CObjDetector;
class CReID;
class CVideoWriter;

// Class for AI-based analysis library
// This class serves as the interface for the AI-based analysis library
// This class is built in a form of pipeline, which contains several analysis tasks
class IAIANALYSISLIB_API CAIAnalysis
{
public:
	CAIAnalysis(const S_AnalysisParam& stParam);
	~CAIAnalysis();

	// Run the given analysis task
	// @param[in] eTaskType: the type of analysis task
	// @param[in] cvBGRFrame: the input BGR format frame
	// @return true if the task is run successfully, otherwise false
	bool RunTask(const E_AnalysisTaskType& eTaskType, const cv::Mat& cvBGRFrame);

	// Begin to write task results to video, given the result type
	// @param[in] eWriteTaskType: the type of result to write to video
	// @param[in] sVideoPath: the path of the video to write. The file extension should be "mp4" or "avi".
	// @param[in] nFPS: the FPS of the video to write
	// @param[in] nW: the width of the video to write
	// @param[in] nH: the height of the video to write
	// @return true if the video writer is opened successfully, otherwise false
	// [Note] - The relevant video codec will be determined by the file extension automatically.
	//        - The task result will be written to video from the point of calling this function.
	bool BeginVideoWriter(const E_AnalysisTaskType& eWriteTaskType, const std::string& sVideoPath, int nFPS, int nW, int nH);

	// End to write task results to video
	// @return true if the video writer is closed successfully, otherwise false
	bool EndVideoWriter();

	// Get the detection result
	// @return the detection result
	const ObjBoxArr*	GetDetectionResult() const;

	// Get the re-identification result
	// @return the re-identification result
	const ReIDResArr*	GetReIDResult() const;

	// Check if the analysis library is valid
	// @return true if the analysis library is valid, otherwise false
	const bool			IsValid() const;

	// Draw the result on the given frame for the given task type.
	// @param[in] eDrawTaskType: the type of analysis task
	// @param[in] bCheckResultExistence: true if the result existence should be checked, otherwise false
	//            If the result existence is checked, the result will be drawn only if the result exists. otherwise return false.
	// @param[in/out] cvBGRFrame: the input BGR format frame
	bool DrawResult(const E_AnalysisTaskType& eDrawTaskType, bool bCheckResultExistence, cv::Mat& cvBGRFrame);

private:
	bool Init();
	bool InitObjDetector();
	bool InitReID();
	bool InitVideoWriter();

	void Release();

private: 
	// Run the detection task
	// @param[in] cvBGRFrame: the input BGR format frame
	// @return true if the task is run successfully, otherwise false
	inline bool RunDetection(const cv::Mat& cvBGRFrame);
	
	// Run the re-identification task
	// @param[in] cvBGRFrame: the input BGR format frame
	// @return true if the task is run successfully, otherwise false
	inline bool RunReID(const cv::Mat& cvBGRFrame);

	// Run the registration task
	// @param[in] cvBGRFrame: the input BGR format frame
	// @return true if the task is run successfully, otherwise false
	inline bool RunRegistration(const cv::Mat& cvBGRFrame);


	// Write the task result to video
	// @param[in] cvBGRFrame: the input BGR format frame
	// @return true if the task result is written to video successfully, otherwise false
	inline bool WriteResultVideo(const cv::Mat& cvBGRFrame);
	
private:
	bool 				m_bValid;			// true if the analysis library is valid	

	CVideoWriter		*m_pVideoWriter;	// Video writer
	CObjDetector		*m_pObjDetector;	// Object detector
	CReID				*m_pReID;			// Re-identification
	S_AnalysisParam		m_stParam;			// Analysis parameters
	
	E_AnalysisTaskType	m_eWriteResultType;	// The type of result to write to video
};