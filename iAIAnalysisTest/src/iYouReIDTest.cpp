#include "CORTYouReID.h"
#include "iAIAnalysisTest.h"



void TestYouReID(int argc, char** argv)
{
	ReIDNetConfig stReIDNetCfg = {
		10,
		0.5f,
		"models/youreid/youtu_reid_baseline_medium.onnx",
	};

	NetDetailsConfig stYouReIDNetDetailsCfg = {
		-1,
		(double)0.406f * 255,
		(double)0.456f * 255,
		(double)0.485f * 255,
		(double)1.0 / ((double)0.225f * 255),
		(double)1.0 / ((double)0.224f * 255),
		(double)1.0 / ((double)0.229f * 255),
	};

	CYouReID net(stReIDNetCfg, stYouReIDNetDetailsCfg);

	std::string sQImgPath = "assets/images/reid/query/0030_c1_f0056923.jpg";
	std::string sGalleryFolder = "assets/images/reid/gallery/";

	
	// Read the query image
	cv::Mat cvQImg = cv::imread(sQImgPath);

	// Read the gallery images under the given folder path using glob
	std::vector<cv::Mat>		cvGImgs;
	std::vector<std::string>	vGImgNames;

	cv::glob(sGalleryFolder, vGImgNames);
	for (auto& sGalleryImgName : vGImgNames)
	{
		cvGImgs.push_back(cv::imread(sGalleryImgName));
	}

	// Stress test for 100 times
	for(int i = 0; i< 100; i++)
	{
		// Perform ReID
		const ReIDResArr& res = net.ReID(cvQImg, cvGImgs);
	}

	// Visualise the ReID results
	cv::Mat cvVisImg = net.Visualise(cvQImg, cvGImgs, 128, 256);

	cv::namedWindow("test", cv::WINDOW_NORMAL);
	cv::imshow("test", cvVisImg);
	cv::waitKey(0);
	cv::destroyAllWindows();
}
