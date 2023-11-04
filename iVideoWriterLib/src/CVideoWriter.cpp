#include "CVideoWriter.h"


CVideoWriter::CVideoWriter(const E_WriterEngineType eWriter /*= E_WriterEngineType::eWETOpenCV*/)
	: m_eWriterEngineType(eWriter)
	, m_eEncoderType(E_EncoderType::eETMPEG4)
	, m_bValid(false)
{

}

CVideoWriter::~CVideoWriter()
{
	Release();
}

// Initialise and open video writer
// @param[in] sVideoPath: video path to save. The file extension should be "mp4" or "avi".
// @param[in] nWidth: video width
// @param[in] nHeight: video height
// @param[in] nFPS: video fps
// @return true if success, otherwise false
// [Note] The relevant video codec will be determined by the file extension automatically.
bool CVideoWriter::Open(const std::string& sVideoPath, int nWidth, int nHeight, int nFPS)
{
	Release();
	
	try
	{
		// Get the video codec from the file extension
		std::string sExt = sVideoPath.substr(sVideoPath.find_last_of(".") + 1);
		// Convert to lower case
		std::transform(sExt.begin(), sExt.end(), sExt.begin(), ::tolower);

		if (sExt == "mp4")
			m_eEncoderType = E_EncoderType::eETMPEG4; // or E_EncoderType::eETMPEG4
		else if (sExt == "avi")
			m_eEncoderType = E_EncoderType::eETMJPEG;
		else
		{
			return false;
		}

		if(m_eWriterEngineType == E_WriterEngineType::eWETOpenCV)
			m_bValid = OpenCVVideoWriter(sVideoPath, nWidth, nHeight, nFPS);
		else
		{
			return false;
		}
	}
	catch (cv::Exception& e)
	{
		std::cout << "CVideoWriter::Open: " << e.what() << std::endl;
		return false;
	}
	catch (std::exception& e)
	{
		std::cout << "CVideoWriter::Open: " << e.what() << std::endl;
		return false;
	}
	catch (...)
	{
		std::cout << "CVideoWriter::Open: Unknown exception" << std::endl;
		return false;
	}
	

	return m_bValid;
}

// Write frame to video
// @param[in] frame: frame to write
// @return true if success, otherwise false
bool CVideoWriter::WriteFrame(const cv::Mat& frame)
{
	if(!m_bValid)
		return false;


	try
	{
		if(m_eWriterEngineType == E_WriterEngineType::eWETOpenCV)
			m_cvVideoWriter.write(frame);
		else
		{
			return false;
		}
	}
	catch (cv::Exception& e)
	{
		std::cout << "CVideoWriter::WriteFrame: " << e.what() << std::endl;
		return false;
	}

	return true;

}

void CVideoWriter::Release()
{
	if(m_cvVideoWriter.isOpened())
		m_cvVideoWriter.release();

	m_bValid = false;
}

// Create and open OpenCV video writer
// @param[in] sVideoPath: video path to save
// @param[in] nWidth: video width
// @param[in] nHeight: video height
// @param[in] nFPS: video fps
// @return true if success, otherwise false
bool CVideoWriter::OpenCVVideoWriter(const std::string& sVideoPath, int nWidth, int nHeight, int nFPS)
{
	// Create OpenCV video writer
	int nFourCC = 0;
	if (m_eEncoderType == E_EncoderType::eETH264)
		nFourCC = cv::VideoWriter::fourcc('H', '2', '6', '4');
	else if(m_eEncoderType == E_EncoderType::eETMPEG4)
		nFourCC = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
	else if (m_eEncoderType == E_EncoderType::eETMJPEG)
		nFourCC = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
	else
		return false;

	m_cvVideoWriter = cv::VideoWriter(sVideoPath, nFourCC, nFPS, cv::Size(nWidth, nHeight));

	return m_cvVideoWriter.isOpened();
}

