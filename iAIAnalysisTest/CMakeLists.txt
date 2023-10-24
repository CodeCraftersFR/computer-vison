project(iAIDetectorTest)

# Glob all .cpp and .h files under src directory
file(GLOB_RECURSE SOURCES "src/*.cpp" "include/*.h")

# Add executable target
add_executable(${PROJECT_NAME} ${SOURCES})

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 20)
endif()

# Set paths of OpenCV headers and libraries
set(OpenCV_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/lib/opencv/include")
set(OpenCV_LIB_DIR "${CMAKE_SOURCE_DIR}/lib/opencv/x64/vc16/lib")
set(OpenCV_LIBS_DEBUG "${OpenCV_LIB_DIR}/opencv_world480d.lib")
set(OpenCV_LIBS_RELEASE "${OpenCV_LIB_DIR}/opencv_world480.lib")


# Set paths of ONNXRUNTIME headers
set(ORT_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/lib/onnxruntime/include")

# Set paths of iAIDetectorLib headers and libraries
set(iAIDetectorLib_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/iAIDetectorLib/include")
set(iAIDetectorLib_LIB_DIR "${CMAKE_SOURCE_DIR}/lib")
set(iAIDetectorLib_LIBS_DEBUG "${iAIDetectorLib_LIB_DIR}/iAIDetectorLibd.lib")
set(iAIDetectorLib_LIBS_RELEASE "${iAIDetectorLib_LIB_DIR}/iAIDetectorLib.lib")

include_directories(
    include
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIR}
    ${ORT_INCLUDE_DIR}
    ${iAIDetectorLib_INCLUDE_DIR}
)

target_link_libraries(${PROJECT_NAME} 
	debug ${OpenCV_LIBS_DEBUG} ${iAIDetectorLib_LIBS_DEBUG}
	optimized ${OpenCV_LIBS_RELEASE} ${iAIDetectorLib_LIBS_RELEASE}
)


# Add the suffix of d to the debug mode library
set_target_properties(${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX d)

# Change the output directory of the executable file
set_target_properties(${PROJECT_NAME} PROPERTIES 
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_SOURCE_DIR}/bin/debug
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_SOURCE_DIR}/bin/release
)

# Print the string to note the completion of the build
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
	COMMAND ${CMAKE_COMMAND} -E echo "Build-${PROJECT_NAME} complete!"
)
