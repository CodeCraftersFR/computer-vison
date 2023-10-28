#include "CObjDetector.h"

CObjDetector::CObjDetector(const ObjDetNetConfig& stObjDetNetConfig)
	: m_stObjDetConfig(stObjDetNetConfig)
{

}

CObjDetector::~CObjDetector()
{
	m_vObjBoxes.clear();
	m_vClsNames.clear();
}

void CObjDetector::DrawBBox(cv::Mat& cvFrame, const ObjBoxArr& vBoxes)
{
	for (auto& bbox : vBoxes)
	{
		int nX1 = int(bbox.fX1), nY1 = int(bbox.fY1);
		int nX2 = int(bbox.fX2), nY2 = int(bbox.fY2);
		cv::rectangle(cvFrame, cv::Point(nX1, nY1), cv::Point(nX2, nY2), cv::Scalar(0, 255, 0), 2);

		cv::putText(cvFrame, m_vClsNames.at(bbox.nClassID),
			cv::Point(nX1, nY1 - 3), cv::FONT_ITALIC,
			0.8, cv::Scalar(255, 255, 255), 2);

		cv::putText(cvFrame, std::to_string(bbox.fScore),
			cv::Point(nX1, nY1 + 30), cv::FONT_ITALIC,
			0.8, cv::Scalar(255, 255, 0), 2);
	}
}


// Perform non-maximum suppression
// @param vDetObj: detected objects before NMS
// @return vDetObj: detected objects after NMS
void CObjDetector::NMSBoxes(ObjBoxArr* pvDetObj)
{
	// Sort the detected objects by score
	sort(pvDetObj->begin(), pvDetObj->end(), [](ObjBBox a, ObjBBox b) { return a.fScore > b.fScore; });

	// Calculate the area of each detected object
	std::vector<float> vArea(pvDetObj->size());
	for (int i = 0; i < int(pvDetObj->size()); ++i)
	{
		vArea[i] = (pvDetObj->at(i).fX2 - pvDetObj->at(i).fX1 + 1)
			* (pvDetObj->at(i).fY2 - pvDetObj->at(i).fY1 + 1);
	}

	std::vector<bool> vIsSupressed(pvDetObj->size(), false);
	for (int i = 0; i < int(pvDetObj->size()); ++i)
	{
		if (vIsSupressed[i]) { continue; }
		for (int j = i + 1; j < int(pvDetObj->size()); ++j)
		{
			if (vIsSupressed[j]) { continue; }
			float fX1 = _MAX(pvDetObj->at(i).fX1, pvDetObj->at(j).fX1);
			float fY1 = _MAX(pvDetObj->at(i).fY1, pvDetObj->at(j).fY1);
			float fX2 = _MIN(pvDetObj->at(i).fX2, pvDetObj->at(j).fX2);
			float fY2 = _MIN(pvDetObj->at(i).fY2, pvDetObj->at(j).fY2);

			float fW = _MAX(float(0), fX2 - fX1 + 1);
			float fH = _MAX(float(0), fY2 - fY1 + 1);
			float fInterArea = fW * fH;
			float fIOU = fInterArea / (vArea[i] + vArea[j] - fInterArea);

			if (fIOU >= m_stObjDetConfig.fNMSThresh)
			{
				vIsSupressed[j] = true;
			}
		}
	}
	// return post_nms;
	int idx_t = 0;
	pvDetObj->erase(remove_if(pvDetObj->begin(), pvDetObj->end(), [&idx_t, &vIsSupressed](const ObjBBox& f) { return vIsSupressed[idx_t++]; }), pvDetObj->end());
}