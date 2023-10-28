#pragma once
#include "core_define.h"

#ifdef _IAICOMMONLIB_
#define IAICOMMONLIB_API _EXPORT_
#else
#define IAICOMMONLIB_API _IMPORT_
#endif
