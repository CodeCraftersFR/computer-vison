#include "CORTTorchReID.h"

CORTTorchReID::CORTTorchReID(const ReIDNetConfig& stReIDNetConfig, const NetDetailsConfig& stNetDetailsConfig)
	: CReID(stReIDNetConfig)
	, CORTInferer(stNetDetailsConfig)
{
	m_bValid = (ReadModel(stReIDNetConfig.sModelPath,
		"", "torchreid-onnx") && Validate());
}

CORTTorchReID::~CORTTorchReID()
{

}

// Perform ReID between the query image and the gallery images
// @param[in] cvQueryImg: single query image
// @param[in] cvGalleryImgs: multiple gallery images
// @return: ReID results
const ReIDResArr& CORTTorchReID::ReID(const cv::Mat& cvQueryImg, const std::vector<cv::Mat>& cvGalleryImgs)
{
	m_vReIDRes.clear();

	// Extract the embedding feature of the query image
	std::vector<float> vQueryFeature;
	if (!ExtractFeature(cvQueryImg, vQueryFeature))
	{
		return m_vReIDRes;
	}

	// Extract the embedding features of the gallery images
	std::vector<std::vector<float>> vGalleryFeatures;
	for (auto& cvGalleryImg : cvGalleryImgs)
	{
		std::vector<float> vGalleryFeature;
		if (!ExtractFeature(cvGalleryImg, vGalleryFeature))
		{
			return m_vReIDRes;
		}
		vGalleryFeatures.push_back(vGalleryFeature);
	}

	// Calculate Top K results
	CalculateTopK(vQueryFeature, vGalleryFeatures, E_SimilarityMetric::COSINE);

	// Return the ReID results
	return m_vReIDRes;
}

// Perform ReID between the query embedding feature and the gallery images
// @param[in] vQueryFeature: query embedding feature
// @param[in] cvGalleryImgs: multiple gallery images
// @return: ReID results
const ReIDResArr& CORTTorchReID::ReID(const std::vector<float>& vQueryFeature, const std::vector<cv::Mat>& cvGalleryImgs)
{
	m_vReIDRes.clear();
	if (vQueryFeature.size() == 0)
		return m_vReIDRes;

	// Extract the embedding features of the gallery images
	std::vector<std::vector<float>> vGalleryFeatures;
	for (auto& cvGalleryImg : cvGalleryImgs)
	{
		std::vector<float> vGalleryFeature;
		if (!ExtractFeature(cvGalleryImg, vGalleryFeature))
		{
			return m_vReIDRes;
		}
		vGalleryFeatures.push_back(vGalleryFeature);
	}

	// Calculate Top K results
	CalculateTopK(vQueryFeature, vGalleryFeatures, E_SimilarityMetric::COSINE);

	// Return the ReID results
	return m_vReIDRes;
}

// Extract the feature vector from the input image
// @param[in] cvImg: input image
// @param[out] vFeature: extracted feature vector
// @return: true if the feature is successfully extracted, false otherwise
const bool CORTTorchReID::ExtractFeature(const cv::Mat& cvImg, std::vector<float>& vFeature)
{
	if (!CORTInferer::Inference(cvImg, &vFeature))
	{
		return false;
	}

	return true;
}

// Preprocess the input image
// @param[in] cvImg: input image to be preprocessed
// @param[out] cvProcImg: preprocessed image
void CORTTorchReID::PreProcess(const cv::Mat& cvImg, cv::Mat& cvProcImg)
{
	CORTInferer::PreProcessCore(cvImg, true, cvProcImg);
}

// Postprocess the output tensor
// @param[in] cvOrgImgSize: original image size
// @param[in] pTensorData: output tensor data to be postprocessed
// @param[out] pPostProcessData: postprocessed data
// [Note] - The normalised feature vector is stored in pPostProcessData
void CORTTorchReID::PostProcess(const cv::Size& cvOrgImgSize, const void* pTensorData, void* pPostProcessData)
{
	if (!pPostProcessData)
		return;

	const float* pData = (const float*)CORTInferer::PostProcessCore(pTensorData);
	if(!pData)
		return;

	// Initialise the original feature vector using the same buffer of pData
	std::vector<float> vOrgFeature(pData, pData + m_nNetProposals);

	// Normalise the feature vector
	std::vector<float>* pvFeature = (std::vector<float>*)pPostProcessData;
	Normalisation(vOrgFeature, *pvFeature);
}

