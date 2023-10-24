# AIAnalysis Libraries
- [Description](#description)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)

## Description
C++ Libraries for AI based CCTV footage analysis. The features are as follows:
- Object Detection using YoloV7 that can detect 80 different objects
- ReID using torchreid that can reidentify people in the footage
- Deep learning models are inferenced using [ONNXRUNTIME](https://onnxruntime.ai/) and [OpenCV](https://opencv.org/)
- The libraries are built using [CMake](https://cmake.org/) for esay installation and usage


## Project Structure
| Name | Description | Development Status |
| --- | --- | --- |
| iAIDetectorLib | Object Detection Dynamic Library | completed |
| iAIReIDLib| ReID Dynamic Library | not started |
| iAIAnalysisLib | AI Analysis Dynamic Library that uses the above two libraries | not started |
| iAIAnalysisTest | Test Application for the above libraries | on progress |
| include | Header files for iAIAnalysisLib | on progress |
| lib | 3rd party libraries that are used in the project | on progress |
| models | Pretrained models for the libraries | on progress |
| assets | Sample images and videos for testing the libraries | on progress |

## Requirements
- [CMake 3+](https://cmake.org/)
- [OpenCV 4+](https://opencv.org/)
- [ONNXRUNTIME 1.16+](https://onnxruntime.ai/)
- MSVC 2022 (Windows) or GCC (Linux)

## Installation
```bash
cmake --build . --config Release
```

## Usage
[This section will be updated after completing `iAIAnalysisLib`]

