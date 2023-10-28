#include "CORTYoloV7.h"
#include "iAIAnalysisTest.h"

void TestYoloV7(int argc, char** argv)
{
	
	ObjDetNetConfig stObjDetNetCfg = { 
		0.3, 
		0.5, 
		"models/yolo7/yolov7_640x640.onnx",
		"models/yolo7/class.names"
	};

	NetDetailsConfig stYolo7NetDetailsCfg = {
		-1,
		0, 0, 0,
		(double)1.0 / 255,
		(double)1.0 / 255,
		(double)1.0 / 255
	};

	CORTYoloV7 net(stObjDetNetCfg, stYolo7NetDetailsCfg);

	std::string imgpath = "assets/images/bus.jpg";
	cv::Mat srcimg = cv::imread(imgpath);
	const ObjBoxArr& vBoxArr = net.Detect(srcimg);
	
	net.DrawBBox(srcimg, vBoxArr);
	
	cv::namedWindow("test", cv::WINDOW_NORMAL);
	cv::imshow("test", srcimg);
	cv::waitKey(0);
	cv::destroyAllWindows();
}
