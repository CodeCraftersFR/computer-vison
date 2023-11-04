#pragma once
#include "core_define.h"
#include "macro_define.h"

// Enum type that defines the encoder type supported
typedef enum _E_ENCODER_TYPE
{
	eETUnknown = -1,	// unknown codec type
	eETH264,			// H.264 codec to write mp4 file. Requires 3rd party x264 library
	eETMPEG4,			// MPEG-4 codec to write mp4 file
	eETMJPEG,			// Motion JPEG codec to write avi file
	eETCnt				// count of codec type supported
}E_EncoderType;


// Enum type that defines the writer engine type supported
// Currently, only OpenCV writer engine is supported. FFmpeg writer engine can be supported as needed in the future.
typedef enum _E_WRITER_ENGINE_TYPE
{
	eWETUnknown = -1,	// unknown writer engine type
	eWETOpenCV,			// OpenCV writer engine
	eWETCnt				// count of writer engine type supported
}E_WriterEngineType;