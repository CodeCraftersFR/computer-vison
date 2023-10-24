#pragma once
#include <onnxruntime_cxx_api.h>

// Class to handle the ONNX runtime session params
class CORTPars
{
public:
	CORTPars() {};
	~CORTPars() {};

	Ort::Env env{ nullptr };								// ONNX runtime environment
	Ort::SessionOptions sessionOptions{ nullptr };			// ONNX runtime session options
	Ort::Session session{ nullptr };						// ONNX runtime session
	Ort::AllocatorWithDefaultOptions allocator;				// ONNX runtime allocator
	std::vector<Ort::AllocatedStringPtr> m_vInputNamesPtr;	// ONNX runtime input names
	std::vector<Ort::AllocatedStringPtr> m_vOutputNamesPtr; // ONNX runtime output names
	std::vector<const char*> m_vInputNames;				// ONNX runtime input names
	std::vector<const char*> m_vOutputNames;				// ONNX runtime output names
};
