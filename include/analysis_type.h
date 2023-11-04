#pragma once
#include <core_type.h>
#include <analysis_define.h>

// Enum type that defines the type of analysis task
typedef enum _E_ANALYSIS_TASK_TYPE
{
	eAttUnknown = -1,		// unknown task
	eAttPersonDetection,	// person detection
	eAttPersonRegister,		// person registration for re-identification
	eAttPersonReID,			// person re-identification
	eAttCount				// total number of tasks supported
}E_AnalysisTaskType;

// Enum type that defines the algorithm mode for person detection
// Currently only YoloV7 is supported. More modes will be added in the future as project goes on.
typedef enum _E_DETECTION_MODE
{
	eDMUnknown = -1,		// unknown mode
	eDMYoloV7,				// YoloV7 mode
	eDMCount				// total number of modes supported
}E_DetectionMode;


// Enum type that defines the algorithm mode for re-identification
// Currently only YouReID and TorchReID are supported. More modes will be added in the future as project goes on.
typedef enum _E_REID_MODE
{
	eRmUnknown = -1,		// unknown mode
	eRmYouReID,				// YouReID mode
	eRmTorchReID,			// TorchReID mode
	eRmCount				// total number of modes supported
}E_ReIDMode;

// Enum type that defines the device type where the analysis task is running
// Currently only CPU is supported. More devices will be added in the future as project goes on.
typedef enum _E_DEVICE_TYPE
{
	eDtUnknown = -1,		// unknown device
	eDtCPU,					// CPU
	eDtCount				// total number of devices supported
}E_DeviceType;

// Enum type that defines the deep learning inference runtime type
// Currently only onnx runtime is supported. More runtimes will be added in the future as project goes on.
typedef enum _E_INFERENCE_RUNTIME_TYPE
{
	eIrtUnknown = -1,		// unknown runtime
	eIrtOnnx,				// onnx runtime
	eIrtCount				// total number of runtimes supported
}E_InferenceRuntimeType;


// Structure that defines the parameters for CAIAnalysisLib
typedef struct _S_ANALYSIS_PARAM
{
	E_DeviceType eDeviceType;				// device type
	E_InferenceRuntimeType eRuntimeType;	// inference runtime type

	E_DetectionMode eDetectionMode;			// object detection mode
	float fDetConfThresh;					// object detection confidence threshold

	E_ReIDMode eReIDMode;					// re-id mode
	float fReIDConfThresh;					// re-id confidence threshold
	int nReIDTopK;							// re-id top k

	_S_ANALYSIS_PARAM(
		E_DeviceType _eDeviceType				= E_DeviceType::eDtCPU, 
		E_InferenceRuntimeType _eRuntimeType	= E_InferenceRuntimeType::eIrtOnnx, 
		E_DetectionMode _eDetectionMode			= E_DetectionMode::eDMYoloV7, 
		float _fDetConfThresh					= 0.5f,
		E_ReIDMode _eReIDMode					= E_ReIDMode::eRmYouReID, 
		float _fReIDConfThresh					= 0.5f,
		int _nReIDTopK							= 5)
	{
		eDeviceType = _eDeviceType;
		eRuntimeType = _eRuntimeType;
		eDetectionMode = _eDetectionMode;
		fDetConfThresh = _fDetConfThresh;
		eReIDMode = _eReIDMode;
		fReIDConfThresh = _fReIDConfThresh;
		nReIDTopK = _nReIDTopK;
	}
}S_AnalysisParam;
