# AIAnalysis Libraries
- [Description](#description)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Build](#build)
- [Usage](#usage)

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
| iAIDetectorLib | Object Detection Dynamic Library that was built using iAICommonLib and YoloV7 | **`completed`** |
| iAIReIDLib| ReID Dynamic Library that was built using iAICommonLib and torchreid/fast-reid/youreid | `on progress` |
| iAIAnalysisLib | AI Analysis Dynamic Library that uses the above two libraries | not started |
| iAIAnalysisTest | Test Application for the above libraries | `on progress` |
| include | Header files for iAIAnalysisLib | `on progress` |
| lib | 3rd party libraries that are used in the project | `on progress` |
| models | Pretrained models for the libraries | `on progress` |
| assets | Sample images and videos for testing the libraries | `on progress` |

### Project Dependencie Structure
| Name | Dependencies |
| --- | --- |
| iAICommonLib | `OpenCV`, `ONNXRUNTIME` |
| iAIDetectorLib | `iAICommonLib`, `YoloV7` |
| iAIReIDLib | `iAICommonLib`, `torchreid`/`fast-reid`/`youreid` |
| iAIAnalysisLib | `iAICommonLib`, `iAIDetectorLib`, `iAIReIDLib` |
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

## Usage
[This section will be updated after completing `iAIAnalysisLib`]

