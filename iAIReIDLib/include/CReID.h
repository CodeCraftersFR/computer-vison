#pragma once
#include "type_define.h"
#include <opencv2/opencv.hpp>

// Abstract base class for ReID(Re-identification)
// All the ReID classes should inherit from this class
class IAIREIDLIB_API CReID
{
public:
	typedef enum _E_SIMILARITY_METRIC
	{
		COSINE = 0,
		EUCLIDEAN
	}E_SimilarityMetric;

	CReID(const ReIDNetConfig& stReIDNetConfig);
	virtual ~CReID();

	// Perform ReID between the query image and the gallery images
	// @param[in] cvQueryImg: single query image
	// @param[in] cvGalleryImgs: multiple gallery images
	// @return: ReID results
	virtual const ReIDResArr& ReID(const cv::Mat& cvQueryImg, const std::vector<cv::Mat>& cvGalleryImgs) = 0;

	// Perform ReID between the query embedding feature and the gallery images
	// @param[in] vQueryFeature: query embedding feature
	// @param[in] cvGalleryImgs: multiple gallery images
	// @return: ReID results
	virtual const ReIDResArr& ReID(const std::vector<float>& vQueryFeature, const std::vector<cv::Mat>& cvGalleryImgs) = 0;

	// Extract the feature vector from the input image
	// @param[in] cvImg: input image
	// @param[out] vFeature: extracted feature vector
	// @return: true if the feature is successfully extracted, false otherwise
	virtual const bool ExtractFeature(const cv::Mat& cvImg, std::vector<float>& vFeature) = 0;

	// Visualise the ReID results
	// @param[in] cvQueryImg: query image
	// @param[in] cvGalleryImgs: gallery images
	// @param[in] vReIDRes: ReID results
	// @param[in] nResizeW: width of the resized image
	// @param[in] nResizeH: height of the resized image
	// @return: visualised image
	// [note] - This is only for verifying the result in test mode. Do not use it in the production environment.
	virtual cv::Mat Visualise(const cv::Mat& cvQueryImg, const std::vector<cv::Mat>& cvGalleryImgs, int nResizeW, int nResizeH);
protected:

	// Normalise the feature vector
	// @param[in] vOrgFeature: original feature vector
	// @param[out] vNorFeature: normalised feature vector
	// [Note] - The default implementation is L2 normalisation. 
	//        - If you want to use other normalisation methods, this function should be overridden.
	virtual void Normalisation(const std::vector<float>& vOrgFeature, std::vector<float>& vNorFeature);

	// Calculate the similarity between the query feature and the gallery features and store the top K results in m_vReIDRes
	// @param[in] vQueryFeature: query feature vector
	// @param[in] vGalleryFeatures: gallery feature vectors
	// @param[in] eMode: similarity mode. COSINE: cosine similarity; EUCLIDEAN: Euclidean distance
	virtual void CalculateTopK(const std::vector<float>& vQueryFeature, 
		const std::vector<std::vector<float>>& vGalleryFeatures,
		const E_SimilarityMetric& eMode = E_SimilarityMetric::COSINE);


private:
	// Calculate the cosine similarity between two vectors
	// @param[in] vFeature1: normalised feature vector 1
	// @param[in] vFeature2: normalised feature vector 2
	// @return: cosine similarity
	float CosineSimilarity(const std::vector<float>& vFeature1, const std::vector<float>& vFeature2);

	// Calculate the Euclidean distance between two vectors
	// @param[in] vFeature1: normalised feature vector 1
	// @param[in] vFeature2: normalised feature vector 2
	// @return: Euclidean distance
	float EuclideanDistance(const std::vector<float>& vFeature1, const std::vector<float>& vFeature2);

protected:
	ReIDNetConfig		m_stReIDNetConfig;		// ReID network configuration
	ReIDResArr			m_vReIDRes;				// ReID results
};