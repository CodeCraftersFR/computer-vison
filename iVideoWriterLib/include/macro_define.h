#pragma once
#include "core_define.h"

#ifdef _IVIDEOWRITERLIB_
#define IVIDEOWRITERLIB_API _EXPORT_
#else
#define IVIDEOWRITERLIB_API _IMPORT_
#endif
