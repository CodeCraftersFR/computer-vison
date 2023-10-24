#pragma once
#include "core_define.h"

#ifdef _IAIDETECTORLIB_
#define IAIDETECTORLIB_API _EXPORT_
#else
#define IAIDETECTORLIB_API _IMPORT_
#endif
