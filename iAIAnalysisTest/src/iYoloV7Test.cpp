#include "CYoloV7.h"
#include "iAIAnalysisTest.h"

void TestYoloV7()
{
	
	YoloNetConfig YOLOV7_nets = { 
		0.3, 
		0.5, 
		"models/yolov7_640x640.onnx",
		"models/class.names"
	};

	CYoloV7 net(YOLOV7_nets);
	std::string imgpath = "assets/images/bus.jpg";
	cv::Mat srcimg = cv::imread(imgpath);
	net.Detect(srcimg);

	
	cv::namedWindow("test", cv::WINDOW_NORMAL);
	cv::imshow("test", srcimg);
	cv::waitKey(0);
	cv::destroyAllWindows();
}
