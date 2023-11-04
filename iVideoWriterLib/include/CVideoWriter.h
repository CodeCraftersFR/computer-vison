#pragma once
#include <opencv2/opencv.hpp>
#include "type_define.h"

// Class for writing video
class IVIDEOWRITERLIB_API CVideoWriter
{
public:
	// Constructor
	// @param[in] eWriter: writer engine type
	CVideoWriter(const E_WriterEngineType eWriter = E_WriterEngineType::eWETOpenCV);
	~CVideoWriter();

	// Initialise and open video writer
	// @param[in] sVideoPath: video path to save. The file extension should be "mp4" or "avi".
	// @param[in] nWidth: video width
	// @param[in] nHeight: video height
	// @param[in] nFPS: video fps
	// @return true if success, otherwise false
	// [Note] The relevant video codec will be determined by the file extension automatically.
	bool Open(const std::string& sVideoPath, int nWidth, int nHeight, int nFPS);

	// Write frame to video
	// @param[in] frame: frame to write
	// @return true if success, otherwise false
	bool WriteFrame(const cv::Mat& frame);

	// Release video writer
	// This function should be called after all frames are written to flush the buffer
	void Release();

	// Check if video writer is valid
	bool IsValid() const { return m_bValid; }
private:
	// Create and open OpenCV video writer
	// @param[in] sVideoPath: video path to save
	// @param[in] nWidth: video width
	// @param[in] nHeight: video height
	// @param[in] nFPS: video fps
	// @return true if success, otherwise false
	bool OpenCVVideoWriter(const std::string& sVideoPath, int nWidth, int nHeight, int nFPS);

private:
	bool				m_bValid;				// Flag to indicate if video writer is valid
	E_WriterEngineType	m_eWriterEngineType;	// Writer engine type
	E_EncoderType		m_eEncoderType;			// Encoder type

	cv::VideoWriter		m_cvVideoWriter;		// OpenCV video writer
	
};
