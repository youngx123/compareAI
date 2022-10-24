cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(mnn_demo)

message("SOURCE  DIR : ", ${CMAKE_SOURCE_DIR})

# opencv include and libs
set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/Libs/x86/opencv45/lib/cmake/opencv4)
set(OpenCV_INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/Libs/x86/opencv45/include/opencv4)
set(OpenCV_LIBRARIES ${CMAKE_SOURCE_DIR}/Libs/x86/opencv45/lib)
list(APPEND CMAKE_PREFIX_PATH ${CMAKE_SOURCE_DIR}/Libs/x86/opencv45)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${OpenCV_LIBRARIES})
find_package( OpenCV REQUIRED )
message("opencv libs : ", ${OpenCV_LIBS})

# jsoncpp lib
set(jsoncpp_DIRS ${CMAKE_SOURCE_DIR}/Libs/x86/jsonLibs)
include_directories(${jsoncpp_DIRS}/include)
add_library(jsoncpp_lib STATIC IMPORTED)
set_target_properties(jsoncpp_lib PROPERTIES IMPORTED_LOCATION ${jsoncpp_DIRS}/lib/libjsoncpp.a)

# paddle lite 
set(paddle_dir ${CMAKE_SOURCE_DIR}/Libs/x86/paddle-Lite_x86)
include_directories( ${paddle_dir}/include)

add_library(paddle_lite SHARED IMPORTED)
set_target_properties(paddle_lite PROPERTIES IMPORTED_LOCATION ${paddle_dir}/lib/libpaddle_light_api_shared.so)

# find_library(paddle_LIBRARIES paddleLite ${CMAKE_SOURCE_DIR}/Libs/x86/paddle-Lite_x86/lib)

# MESSAGE("paddle-lite_so  :", ${paddle_LIBRARIES})

# IF(NOT paddle_LIBRARIES)
# MESSAGE(FATAL_ERROR "paddle-lite_so not found")
# ENDIF(NOT paddle_LIBRARIES) 

aux_source_directory(. DIR_SRCS)

add_executable(paddle_demo ${DIR_SRCS})
target_link_libraries(paddle_demo paddle_lite  jsoncpp_lib ${OpenCV_LIBS} )
set_property(TARGET paddle_demo PROPERTY CXX_STANDARD 11)
