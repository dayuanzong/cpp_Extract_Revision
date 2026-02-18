cmake_minimum_required(VERSION 3.10)
project(FaceExtractor)

set(CMAKE_CXX_STANDARD 17)

# Paths
# Adjust paths for new structure
# Project root is one level up from core/
get_filename_component(PROJECT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
set(ASSETS_DIR "${PROJECT_ROOT}/assets")

# OpenCV
set(OPENCV_DIR "${ASSETS_DIR}/deps/opencv/build")
include_directories("${OPENCV_DIR}/include")
link_directories("${OPENCV_DIR}/x64/vc15/lib")

# ONNX Runtime
set(ONNXRUNTIME_DIR "${ASSETS_DIR}/deps/onnxruntime")
include_directories("${ONNXRUNTIME_DIR}/include")
link_directories("${ONNXRUNTIME_DIR}/lib")

# Source files
set(LIB_SOURCES 
    "src/S3FDExtractor.cpp"
    "src/SCRFDExtractor.cpp"
    "src/FANExtractor.cpp"
    "src/InsightFaceLandmark.cpp"
    "src/FacePipeline.cpp"
    "src/interface.cpp"
)

# Executable sources (if needed for standalone testing)
set(EXE_SOURCES 
    "src/main.cpp"
    "src/S3FDExtractor.cpp"
    "src/SCRFDExtractor.cpp"
    "src/FANExtractor.cpp"
    "src/InsightFaceLandmark.cpp"
    "src/FacePipeline.cpp"
)

# --- Executable (CLI for debugging) ---
add_executable(FaceExtractor ${EXE_SOURCES})
target_link_libraries(FaceExtractor opencv_world455 onnxruntime)

# --- Shared Library (DLL for Python) ---
add_library(FaceExtractorDLL SHARED ${LIB_SOURCES})
target_compile_definitions(FaceExtractorDLL PRIVATE FACE_EXTRACTOR_EXPORTS)
target_link_libraries(FaceExtractorDLL opencv_world455 onnxruntime)

# --- Post Build Commands (Copy DLLs) ---

# Function to copy DLLs to target directory
function(copy_dlls target_name)
    add_custom_command(TARGET ${target_name} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${OPENCV_DIR}/x64/vc15/bin/opencv_world455.dll"
        $<TARGET_FILE_DIR:${target_name}>
    )
    add_custom_command(TARGET ${target_name} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ONNXRUNTIME_DIR}/lib/onnxruntime.dll"
        $<TARGET_FILE_DIR:${target_name}>
    )
    # Copy CUDA providers if available
    add_custom_command(TARGET ${target_name} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ONNXRUNTIME_DIR}/lib/onnxruntime_providers_cuda.dll"
        $<TARGET_FILE_DIR:${target_name}>
    )
    add_custom_command(TARGET ${target_name} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ONNXRUNTIME_DIR}/lib/onnxruntime_providers_shared.dll"
        $<TARGET_FILE_DIR:${target_name}>
    )
    # Copy TensorRT provider if available (optional)
    if(EXISTS "${ONNXRUNTIME_DIR}/lib/onnxruntime_providers_tensorrt.dll")
        add_custom_command(TARGET ${target_name} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ONNXRUNTIME_DIR}/lib/onnxruntime_providers_tensorrt.dll"
            $<TARGET_FILE_DIR:${target_name}>
        )
    endif()
endfunction()

copy_dlls(FaceExtractor)
copy_dlls(FaceExtractorDLL)
