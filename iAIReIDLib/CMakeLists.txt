project(iAIReIDLib)

#Set the C++ standard to 20
set(CMAKE_CXX_STANDARD 20)

# Set paths of OpenCV headers and libraries
set(OpenCV_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/lib/opencv/include")
set(OpenCV_LIB_DIR "${CMAKE_SOURCE_DIR}/lib/opencv/x64/vc16/lib")
set(OpenCV_LIBS_DEBUG "${OpenCV_LIB_DIR}/opencv_world480d.lib")
set(OpenCV_LIBS_RELEASE "${OpenCV_LIB_DIR}/opencv_world480.lib")


# Set paths of iAICommonLib headers and libraries
set(IAICOMMONLIB_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/iAICommonLib/include")
set(IAICOMMONLIB_LIB_DIR "${CMAKE_SOURCE_DIR}/lib")
set(IAICOMMONLIB_LIBS_DEBUG "${IAICOMMONLIB_LIB_DIR}/iAICommonLibd.lib")
set(IAICOMMONLIB_LIBS_RELEASE "${IAICOMMONLIB_LIB_DIR}/iAICommonLib.lib")

include_directories(
    include
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIR}
    ${IAICOMMONLIB_INCLUDE_DIR}
)

# Glob all .cpp and .h files under src directory
file(GLOB_RECURSE SOURCES "src/*.cpp" "include/*.h")

add_library(${PROJECT_NAME} SHARED ${SOURCES})

# Define preprocessor macros
target_compile_definitions(${PROJECT_NAME} PRIVATE _IAIREIDLIB_)

# Link debug libraries
target_link_libraries(${PROJECT_NAME} 
	debug ${OpenCV_LIBS_DEBUG} 
    debug ${IAICOMMONLIB_LIBS_DEBUG}
)

# Link release libraries
target_link_libraries(${PROJECT_NAME} 
	optimized ${OpenCV_LIBS_RELEASE} 
	optimized ${IAICOMMONLIB_LIBS_RELEASE}
)

# Add the suffix of d to the debug mode library
set_target_properties(${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX d)

# Change the output directory of the library
set_target_properties(${PROJECT_NAME} PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${CMAKE_SOURCE_DIR}/lib
    ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${CMAKE_SOURCE_DIR}/lib
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_SOURCE_DIR}/bin/debug
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_SOURCE_DIR}/bin/release
)

# Set the build dependencies
add_dependencies(${PROJECT_NAME} iAICommonLib)

# Print the string to note the completion of the build
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
	COMMAND ${CMAKE_COMMAND} -E echo "Build-${PROJECT_NAME} complete!"
)