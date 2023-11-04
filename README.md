# AIAnalysis Libraries
- [Description](#description)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Build](#build)
- [Usage](#using-iaianalysislib-library-in-your-project)
    - [Project configuration](#-project-configuration)
    - [Test `Person-ReID` function](#-test-person-reid-function-for-details-please-refer-to-iaianalysistestiaianalysistestcpp)
    - [Test `Person-Detection` function](#-test-person-detection-function-for-details-please-refer-to-iaianalysistestiaianalysistestcpp)
- [Person-ReID Test Result](#person-reid-test-result)
    - [Query Image](#query-image)
    - [Test Video](#test-video)
    - [Result Video](#result-video)
- [Person-Detection Test Result](#person-detection-test-result)
    - [Test Video](#test-video-1)
    - [Result Video](#result-video-1)

- [TODO](#todo)

## Description
C++ Libraries for AI based CCTV footage analysis. The features are as follows:
- Object Detection using YoloV7 that can detect 80 different objects
- ReID using torchreid that can reidentify people in the footage
- Deep learning models are inferenced using [ONNXRUNTIME](https://onnxruntime.ai/) and [OpenCV](https://opencv.org/)
- The libraries are built using [CMake](https://cmake.org/) for esay installation and usage


## Project Structure
### Directory Structure
| Name | Description | Development Status |
| --- | --- | --- |
| iAICommonLib | Common Library for the project | **`completed`** (for ORT) |
| iVideoWriterLib | Video Writer Library | **`completed`**|
| iAIDetectorLib | Object Detection Dynamic Library that was built using iAICommonLib and YoloV7 | **`completed`** |
| iAIReIDLib| ReID Dynamic Library that was built using iAICommonLib and torchreid/fast-reid/youreid | **`completed`**|
| iAIAnalysisLib | AI Analysis Dynamic Library that uses the above two libraries | **`completed`**|
| iAIAnalysisTest | Test Application for the above libraries | **`completed`**|
| include | Header files for iAIAnalysisLib | **`completed`**|
| lib | 3rd party libraries that are used in the project | **`completed`**|
| models | Pretrained models for the libraries | **`completed`**|
| assets | Sample images and videos for testing the libraries | **`completed`**|

### Project Dependencie Structure
| Name | Dependencies |
| --- | --- |
| iAICommonLib | `OpenCV`, `ONNXRUNTIME` |
| iVideoWriterLib | `OpenCV` |
| iAIDetectorLib | `iAICommonLib`, `YoloV7` |
| iAIReIDLib | `iAICommonLib`, `torchreid`/`youreid` |
| iAIAnalysisLib | `iAICommonLib`, `iAIDetectorLib`, `iAIReIDLib`, `iVideoWriterLib` |
| iAIAnalysisTest | `iAICommonLib`, `iAIAnalysisLib` |




## Requirements
- [CMake 3+](https://cmake.org/)
- [OpenCV 4+](https://opencv.org/)
- [ONNXRUNTIME 1.16+](https://onnxruntime.ai/)
- MSVC 2022 (Windows) or GCC (Linux)

## Build
```bash
cmake --build . --config Release
```

## Using `iAIAnalysisLib` library in your project
### - Project configuration
- **Download the latest release from [windows-x64-lib]() and Extract it**
- **Copy the `include`, `lib` and `bin` folders to your project directory**
- **Add the following lines to your `CMakeLists.txt` file. For details, please refer to [iAIAnalysisTest/CMakeLists.txt](iAIAnalysisTest/CMakeLists.txt)** 

```cmake
# Set paths of OpenCV headers and libraries
set(OpenCV_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/lib/opencv/include")
set(OpenCV_LIB_DIR "${CMAKE_SOURCE_DIR}/lib/opencv/x64/vc16/lib")
set(OpenCV_LIBS_DEBUG "${OpenCV_LIB_DIR}/opencv_world480d.lib")
set(OpenCV_LIBS_RELEASE "${OpenCV_LIB_DIR}/opencv_world480.lib")


# Set paths of iAIAnalysisLib libraries
set(iAIAnalysisLib_LIB_DIR "${CMAKE_SOURCE_DIR}/lib")
set(iAIAnalysisLib_LIBS_DEBUG "${iAIAnalysisLib_LIB_DIR}/iAIAnalysisLibd.lib")
set(iAIAnalysisLib_LIBS_RELEASE "${iAIAnalysisLib_LIB_DIR}/iAIAnalysisLib.lib")


include_directories(
    include
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIR}
)

# Link debug libraries
target_link_libraries(${PROJECT_NAME} 
	debug ${OpenCV_LIBS_DEBUG} 
    debug ${iAIAnalysisLib_LIBS_DEBUG}
)

# Link release libraries
target_link_libraries(${PROJECT_NAME} 
	optimized ${OpenCV_LIBS_RELEASE} 
    optimized ${iAIAnalysisLib_LIBS_RELEASE}
)
```

### - Test `Person-ReID` function. For details, please refer to [iAIAnalysisTest/iAIAnalysisTest.cpp](iAIAnalysisTest/iAIAnalysisTest.cpp)

```cpp
```cpp
#include "CAIAnalysis.h"

void TestPersonReID(int argc, char** argv)
{
	// Create AIAnalysis param
	S_AnalysisParam stParam{
		E_DeviceType::eDtCPU,			// Run on CPU
		E_InferenceRuntimeType::eIrtOnnx,	// Use ONNX runtime for inference
		E_DetectionMode::eDMYoloV7,		// Use YoloV7 module for detection
		0.3f,					// Detection threshold
		E_ReIDMode::eRmYouReID,			// Use YouReID module for ReID
		0.5f,					// ReID threshold
		5					// ReID top k to search
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
```
### - Test `Person-Detection` function. For details, please refer to [iAIAnalysisTest/iAIAnalysisTest.cpp](iAIAnalysisTest/iAIAnalysisTest.cpp)
```cpp
#include "CAIAnalysis.h"

void TestPersonDetection(int argc, char** argv)
{
	// Create AIAnalysis param
	S_AnalysisParam stParam{
		E_DeviceType::eDtCPU,			// Run on CPU
		E_InferenceRuntimeType::eIrtOnnx,	// Use ONNX runtime for inference
		E_DetectionMode::eDMYoloV7,		// Use YoloV7 module for detection
		0.3f,					// Detection threshold
		E_ReIDMode::eRmYouReID,			// Use YouReID module for ReID
		0.5f,					// ReID threshold
		5					// ReID top k to search
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
	cAIAnalysis.BeginVideoWriter(E_AnalysisTaskType::eAttPersonDetection, "detection-output.mp4", nFPS, nWidth, nHeight);

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
```

## Person-ReID Test Result
### Query Image

The following image is used as the query image for ReID. The query image is registered to the ReID module before the video analysis starts. It was cropped from the frame around the 10th second of the [test video](assets/videos/test.mp4).
<p align="center">
    <img src="assets/videos/query.bmp" alt="Query Image" width="80"/>
</p>

### Test Video

https://github.com/CodeCraftersFR/computer-vison/assets/videos/test.mp4

### Result Video

The `ReID Confidence Score` is shown on the top left corner of each bounding box.  The bounding box colour is dynamically changed according to the `ReID Confidence Score` as follows:
- The box with the highest score (`1.0`) will have the pure <font color="red"> RED </font> colour.
- The box with the lowest score (`0.0`) will have the pure <font color="green"> GREEN </font> colour.
- The colour of the other boxes will be the mixture of the above two colours.

https://github.com/CodeCraftersFR/computer-vison/assets/112942561/5dc089dc-521a-4fab-b4e4-247200a813c2

## Person-Detection Test Result
### Test Video

This is the same video as the above test video for ReID.

### Result Video

The `Detection Confidence Score` is shown on the top left corner of each bounding box.  The bounding box colour is dynamically changed according to the `Detection Confidence Score` as follows:
- The box with the highest score (`1.0`) will have the pure <font color="red"> RED </font> colour.
- The box with the lowest score (`0.0`) will have the pure <font color="green"> GREEN </font> colour.
- The colour of the other boxes will be the mixture of the above two colours.

<video src="https://github.com/CodeCraftersFR/computer-vison/assets/112942561/8c520e33-8d21-4110-8d3c-fb5acfc74ddc" controls="controls" style="max-width: 730px;">
</video>


## TODO
- [ ] Inference using GPU. Currently, the libraries only support CPU inference.
- [ ] Inference using TensorRT + GPU + FP16. Currently, the libraries only support ONNX + CPU + FP32 inference.
- [ ] Batch inference. Currently, the libraries only support single image/video inference.
