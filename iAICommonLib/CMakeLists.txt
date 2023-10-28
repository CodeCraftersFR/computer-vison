project(iAICommonLib)

#Set the C++ standard to 20
set(CMAKE_CXX_STANDARD 20)

# Set paths of OpenCV headers and libraries
set(OpenCV_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/lib/opencv/include")
set(OpenCV_LIB_DIR "${CMAKE_SOURCE_DIR}/lib/opencv/x64/vc16/lib")
set(OpenCV_LIBS_DEBUG "${OpenCV_LIB_DIR}/opencv_world480d.lib")
set(OpenCV_LIBS_RELEASE "${OpenCV_LIB_DIR}/opencv_world480.lib")


# Set paths of ONNXRUNTIME headers and libraries
set(ORT_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/lib/onnxruntime/include")
set(ORT_LIB_DIR "${CMAKE_SOURCE_DIR}/lib/onnxruntime/lib")
set(ORT_LIBS_DEBUG "${ORT_LIB_DIR}/onnxruntime.lib")
set(ORT_LIBS_RELEASE "${ORT_LIB_DIR}/onnxruntime.lib")


include_directories(
    include
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIR}
    ${ORT_INCLUDE_DIR}
)

# Glob all .cpp and .h files under src directory
file(GLOB_RECURSE SOURCES "src/*.cpp" "include/*.h")

add_library(${PROJECT_NAME} SHARED ${SOURCES})

# Define preprocessor macros
target_compile_definitions(${PROJECT_NAME} PRIVATE _IAICOMMONLIB_)

# Link debug libraries
target_link_libraries(${PROJECT_NAME} 
    debug ${OpenCV_LIBS_DEBUG}
    debug ${ORT_LIBS_DEBUG}
)

# Link release libraries
target_link_libraries(${PROJECT_NAME} 
	optimized ${OpenCV_LIBS_RELEASE}
    optimized ${ORT_LIBS_RELEASE}
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


# Print the string to note the completion of the build
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
	COMMAND ${CMAKE_COMMAND} -E echo "Build-${PROJECT_NAME} complete!"
)