#pragma once

#ifndef _EXPORT_
	#define _EXPORT_ __declspec(dllexport)
#endif

#ifndef _IMPORT_
	#define _IMPORT_ __declspec(dllimport)
#endif


#define _MAX(A, B)						(((A) > (B)) ? (A):(B))
#define _MIN(A, B)						(((A) < (B)) ? (A):(B))
#define _ABS(A)							(((A) < 0) ?   (-(A)):(A))
