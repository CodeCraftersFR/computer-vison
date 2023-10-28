// iAIAnalysisTest.cpp : Defines the entry point for the application.
//
#include "type_define.h"
#include "iAIAnalysisTest.h"

using namespace std;

int main(int argc, char** argv)
{
#if YOLOV7_TEST_MODE
	TestYoloV7(argc, argv);
#endif

#if YOUREID_TEST_MODE
	TestYouReID(argc, argv);
#endif

#if YOUREID_LEGACY_TEST_MODE
	TestYouReIDLegacy(argc, argv);
#endif
	return 0;
}
