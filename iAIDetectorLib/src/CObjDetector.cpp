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

// Set the class names of objects which will be detected by the network
// @param[in] vClsNames2Detect: class names
void CObjDetector::SetClsNames2Detect(const ObjClsArr& vClsNames2Detect)
{
	m_vClsNames2Detect = vClsNames2Detect;
}

// Draw the bounding boxes on the input frame
// @param[in] cvFrame: input frame in BGR format
// @param[in] vBoxes: bounding boxes of detected objects
// @param[in] bDrawClsName: whether to draw the class name of each object
// @param[in] bDrawScore: whether to draw the score of each object
// [Note]: Drawing will be performed on the input frame directly.
void CObjDetector::DrawBBox(cv::Mat& cvFrame, const ObjBoxArr& vBoxes, bool bDrawClsName/* = true*/, bool bDrawScore/* = true*/)
{
	// The box with the highest score will have the pure red colour.
	// The box with the lowest score will have the pure blue colour.
	// The box with the middle score will have the pure green colour.
	// The colour of the other boxes will be the mixture of the above three colours.

	int nThickness = 1;
	int nFont = cv::FONT_HERSHEY_SIMPLEX;

	for (auto& bbox : vBoxes)
	{
		cv::Scalar cvColour = cv::Scalar(0, 255 * (1 - bbox.fScore), 255 * bbox.fScore);
		int nX1 = int(bbox.fX1), nY1 = int(bbox.fY1);
		int nX2 = int(bbox.fX2), nY2 = int(bbox.fY2);
		cv::rectangle(cvFrame, cv::Point(nX1, nY1), cv::Point(nX2, nY2), cvColour, nThickness);

		if(bDrawClsName)
		{
			std::string sLabel = m_vClsNames.at(bbox.nClassID);
			// Get the size of the text
			int nBaseLine = 0;
			cv::Size cvLableSize = cv::getTextSize(sLabel, nFont, (double)nThickness / 3, nThickness, &nBaseLine);

			// Draw the rectangle below the top-left corner of the bounding box
 			cv::rectangle(cvFrame, cv::Point(nX1, nY1),
 				cv::Point(nX1 + cvLableSize.width, nY1 + cvLableSize.height + nBaseLine),
 				cvColour, cv::FILLED);
			
			// Draw the text on the rectangle
 			cv::putText(cvFrame, sLabel, cv::Point(nX1, nY1 + cvLableSize.height + 1),
 				nFont, (double)nThickness / 3, cv::Scalar(255, 255, 255), nThickness, cv::LINE_AA);

		}
		
		if(bDrawScore)
		{
			// Round the score to 2 decimal places
			std::string sLabel = cv::format("%.2f", bbox.fScore);
			// Get the size of the text
			int nBaseLine = 0;
			cv::Size cvLableSize = cv::getTextSize(sLabel, nFont, (double)nThickness / 3, nThickness, &nBaseLine);

			// Draw the rectangle above the top-left corner of the bounding box
			cv::rectangle(cvFrame, cv::Point(nX1, nY1 - cvLableSize.height - nBaseLine),
				cv::Point(nX1 + cvLableSize.width, nY1),
				cvColour, cv::FILLED);

			// Draw the text on the rectangle
			cv::putText(cvFrame, sLabel, cv::Point(nX1, nY1 - nBaseLine/2),
				nFont, (double)nThickness / 3, cv::Scalar(255, 255, 255), nThickness, cv::LINE_AA);
			
		}
	}


}

// Get the class names of objects which can be detected by the network
// @return: class names
const ObjClsArr& CObjDetector::GetClsNames() const
{
	return m_vClsNames;
}

// Get the bounding boxes of detected objects
// @return: bounding boxes
const ObjBoxArr& CObjDetector::GetObjBoxes() const
{
	return m_vObjBoxes;
}

// Perform non-maximum suppression
// @param[in/out] pvBoxes: bounding boxes of detected objects before and after non-maximum suppression
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

// Check if the class name is in the list of class names of objects which can be detected by the network
// @param[in] strClsName: class name
// @return: true if the class name is in the list, false otherwise
bool CObjDetector::IsClsName2Detect(const std::string& strClsName) const
{
	if (m_vClsNames2Detect.size() == 0)
		return true;

	return std::find(m_vClsNames2Detect.begin(), m_vClsNames2Detect.end(), strClsName) != m_vClsNames2Detect.end();
}

// Check if the class ID is in the list of class IDs of objects which can be detected by the network
// @param[in] nClsID: class ID
// @return: true if the class ID is in the list, false otherwise
bool CObjDetector::IsClsID2Detect(const int nClsID) const
{
	if(nClsID < 0 || nClsID >= int(m_vClsNames.size()))
		return false;
	
	const std::string& strClsName = m_vClsNames.at(nClsID);

	return IsClsName2Detect(strClsName);
}
