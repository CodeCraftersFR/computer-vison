// iAIAnalysisTest.cpp : Defines the entry point for the application.
//
#include "type_define.h"
#include "iAIAnalysisTest.h"
#include "CAIAnalysis.h"

using namespace std;

void TestPersonReID(int argc, char** argv)
{
	// Create AIAnalysis param
	S_AnalysisParam stParam{
		E_DeviceType::eDtCPU,				// Run on CPU
		E_InferenceRuntimeType::eIrtOnnx,	// Use ONNX runtime for inference
		E_DetectionMode::eDMYoloV7,			// Use YoloV7 module for detection
		0.3f,								// Detection threshold
		E_ReIDMode::eRmYouReID,				// Use YouReID module for ReID
		0.5f,								// ReID threshold
		5									// ReID top k to search
	};

	// Create AIAnalysis instance with the param
	CAIAnalysis cAIAnalysis(stParam);

	// Read ReID query image
	cv::Mat cvQueryImg = cv::imread("assets/videos/query.bmp");

	// Register ReID query image
	if(!cAIAnalysis.RunTask(E_AnalysisTaskType::eAttPersonRegister, cvQueryImg))
	{
		cout << "Register ReID query image failed!" << endl;
		return;
	}

	
	// Read video
	cv::VideoCapture cvVideo("assets/videos/test.mp4");	
	if(!cvVideo.isOpened())
	{
		cout << "Open video failed!" << endl;
		return;
	}

	
	// Get the video frame size and fps info from video capture
	int nWidth = cvVideo.get(cv::CAP_PROP_FRAME_WIDTH);
	int nHeight = cvVideo.get(cv::CAP_PROP_FRAME_HEIGHT);
	int nFPS = cvVideo.get(cv::CAP_PROP_FPS);

	// Begin video writer
	// Persion ReID result will be written to the "reid-output.mp4" file
	cAIAnalysis.BeginVideoWriter(E_AnalysisTaskType::eAttPersonReID, "reid-output.mp4", nFPS, nWidth, nHeight);

	// Read video frame by frame
	cv::Mat cvFrame;
	cv::namedWindow("Result", cv::WINDOW_NORMAL);
	while(cvVideo.read(cvFrame))
	{
		// ReID
		if(!cAIAnalysis.RunTask(E_AnalysisTaskType::eAttPersonReID, cvFrame))
		{
			cout << "ReID failed!" << endl;
			break;
		}


		// Show result for debug. Not necessary for real application
		cAIAnalysis.DrawResult(E_AnalysisTaskType::eAttPersonReID, false, cvFrame);
		cv::imshow("Result", cvFrame);
		cv::waitKey(1);
	}

	// End video writer. !!! MUST call this function to close the video writer
	cAIAnalysis.EndVideoWriter();


	cv::destroyAllWindows();

}


void TestPersonDetection(int argc, char** argv)
{
	// Create AIAnalysis param
	S_AnalysisParam stParam{
		E_DeviceType::eDtCPU,				// Run on CPU
		E_InferenceRuntimeType::eIrtOnnx,	// Use ONNX runtime for inference
		E_DetectionMode::eDMYoloV7,			// Use YoloV7 module for detection
		0.3f,								// Detection threshold
		E_ReIDMode::eRmYouReID,				// Use YouReID module for ReID
		0.5f,								// ReID threshold
		5									// ReID top k to search
	};

	// Create AIAnalysis instance with the param
	CAIAnalysis cAIAnalysis(stParam);

	// Read video
	cv::VideoCapture cvVideo("assets/videos/test.mp4");
	if (!cvVideo.isOpened())
	{
		cout << "Open video failed!" << endl;
		return;
	}


	// Get the video frame size and fps info from video capture
	int nWidth = cvVideo.get(cv::CAP_PROP_FRAME_WIDTH);
	int nHeight = cvVideo.get(cv::CAP_PROP_FRAME_HEIGHT);
	int nFPS = cvVideo.get(cv::CAP_PROP_FPS);

	// Begin video writer
	// Person detection result will be written to the "detection-ouput.mp4" file
	cAIAnalysis.BeginVideoWriter(E_AnalysisTaskType::eAttPersonDetection, "detection-ouput.mp4", nFPS, nWidth, nHeight);

	// Read video frame by frame
	cv::Mat cvFrame;
	cv::namedWindow("Result", cv::WINDOW_NORMAL);
	while (cvVideo.read(cvFrame))
	{
		// ReID
		if (!cAIAnalysis.RunTask(E_AnalysisTaskType::eAttPersonDetection, cvFrame))
		{
			cout << "ReID failed!" << endl;
			break;
		}


		// Show result for debug. Not necessary for real application
		cAIAnalysis.DrawResult(E_AnalysisTaskType::eAttPersonDetection, false, cvFrame);
		cv::imshow("Result", cvFrame);
		cv::waitKey(1);
	}

	// End video writer. !!! MUST call this function to close the video writer
	cAIAnalysis.EndVideoWriter();


	cv::destroyAllWindows();

}



int main(int argc, char** argv)
{
#if TEST_REID
	TestPersonReID(argc, argv);
#endif

#if TEST_DETECTION
	TestPersonDetection(argc, argv);
#endif

	return 0;
}

