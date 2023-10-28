#pragma once
#include "core_define.h"

#ifdef _IAIREIDLIB_
#define IAIREIDLIB_API _EXPORT_
#else
#define IAIREIDLIB_API _IMPORT_
#endif
