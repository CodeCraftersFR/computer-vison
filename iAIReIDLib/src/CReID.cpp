#include "CReID.h"

CReID::CReID(const ReIDNetConfig& stReIDNetConfig)
	: m_stReIDNetConfig(stReIDNetConfig)
{

}

CReID::~CReID()
{
	m_vReIDRes.clear();
}

// Normalise the feature vector
// @param[in] vOrgFeature: original feature vector
// @param[out] vNorFeature: normalised feature vector
// [Note] - The default implementation is L2 normalisation. 
//        - If you want to use other normalisation methods, this function should be overridden.
void CReID::Normalisation(const std::vector<float>& vOrgFeature, std::vector<float>& vNorFeature)
{
	vNorFeature.clear();

	float fSum = 0.0f;
	for (int i = 0; i < vOrgFeature.size(); i++)
	{
		fSum += vOrgFeature[i] * vOrgFeature[i];
	}
	fSum = sqrt(fSum);
	for (int i = 0; i < vOrgFeature.size(); i++)
	{
		vNorFeature.push_back(vOrgFeature[i] / fSum);
	}
}

// Calculate the similarity between the query feature and the gallery features and store the top K results in m_vReIDRes
// @param[in] vQueryFeature: query feature vector
// @param[in] vGalleryFeatures: gallery feature vectors
// @param[in] eMode: similarity mode. COSINE: cosine similarity; EUCLIDEAN: Euclidean distance
void CReID::CalculateTopK(const std::vector<float>& vQueryFeature, 
	const std::vector<std::vector<float>>& vGalleryFeatures, 
	const E_SimilarityMode& eMode /*= E_SimilarityMode::COSINE*/)
{
	m_vReIDRes.clear();

	std::vector<float>	vSimilarities;
	std::vector<int>	vImgIDs;

	for (int i = 0; i < vGalleryFeatures.size(); i++)
	{
		float fSimilarity = 0.0f;
		if (eMode == E_SimilarityMode::COSINE)
		{
			fSimilarity = CosineSimilarity(vQueryFeature, vGalleryFeatures[i]);
		}
		else if (eMode == E_SimilarityMode::EUCLIDEAN)
		{
			fSimilarity = 1 - EuclideanDistance(vQueryFeature, vGalleryFeatures[i]);
		}
		else
		{
			assert(false);
		}
		vSimilarities.push_back(fSimilarity);
		vImgIDs.push_back(i);
	}

	// Sort the similarities in descending order
	std::sort(vImgIDs.begin(), vImgIDs.end(), [&](int x, int y) {return vSimilarities[x] > vSimilarities[y]; });

	// Store the top K results in m_vReIDRes
	int nLen = _MIN(m_stReIDNetConfig.nTopK, vImgIDs.size());
	for (int i = 0; i < nLen; i++)
	{
		ReIDRes stReIDRes(i, vImgIDs[i], vSimilarities[vImgIDs[i]]);
		m_vReIDRes.push_back(stReIDRes);
	}
}

// Visualise the ReID results
// @param[in] cvQueryImg: query image
// @param[in] cvGalleryImgs: gallery images
// @param[in] vReIDRes: ReID results
// @param[in] nResizeW: width of the resized image
// @param[in] nResizeH: height of the resized image
// @return: visualised image
// [note] - This is only for verifying the result in test mode. Do not use it in the production environment.
cv::Mat CReID::Visualise(const cv::Mat& cvQueryImg, const std::vector<cv::Mat>& cvGalleryImgs, int nResizeW, int nResizeH)
{
	const cv::Size outputSize = cv::Size(nResizeW, nResizeH);
	cv::Mat cvTmpImg, cvVisImg;


	cv::resize(cvQueryImg, cvTmpImg, outputSize);
	cv::copyMakeBorder(cvTmpImg, cvTmpImg, 3, 3, 3, 3, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
	cv::putText(cvTmpImg, "Query", cv::Point(10, 30), cv::FONT_HERSHEY_COMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);


	std::vector<cv::Mat> cvImgs;
	cvImgs.push_back(cvTmpImg);
	for (int i = 0; i < (int)m_vReIDRes.size(); i++)
	{
		resize(cvGalleryImgs[i], cvTmpImg, outputSize);
		cv::copyMakeBorder(cvTmpImg, cvTmpImg, 3, 3, 3, 3, cv::BORDER_CONSTANT, cv::Scalar(255, 255, 255));
		cv::putText(cvTmpImg, "G" + std::to_string(i), cv::Point(10, 30), cv::FONT_HERSHEY_COMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
		cvImgs.push_back(cvTmpImg);
	}
	cv::hconcat(cvImgs, cvVisImg);

	return cvVisImg;
}

// Calculate the cosine similarity between two vectors
// @param[in] vFeature1: normalised feature vector 1
// @param[in] vFeature2: normalised feature vector 2
// @return: cosine similarity
float CReID::CosineSimilarity(const std::vector<float>& vFeature1, const std::vector<float>& vFeature2)
{
	assert(vFeature1.size() == vFeature2.size());
	float fSimilarity = 0.0f;
	for (int i = 0; i < vFeature1.size(); i++)
	{
		fSimilarity += vFeature1[i] * vFeature2[i];
	}

	return fSimilarity;
}


// Calculate the Euclidean distance between two vectors
// @param[in] vFeature1: normalised feature vector 1
// @param[in] vFeature2: normalised feature vector 2
// @return: Euclidean distance
float CReID::EuclideanDistance(const std::vector<float>& vFeature1, const std::vector<float>& vFeature2)
{
	assert(vFeature1.size() == vFeature2.size());

	float fDistance = 0.0f;
	for (int i = 0; i < vFeature1.size(); i++)
	{
		fDistance += (vFeature1[i] - vFeature2[i]) * (vFeature1[i] - vFeature2[i]);
	}

	return sqrt(fDistance);

}
