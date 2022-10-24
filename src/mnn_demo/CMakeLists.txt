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

# MNN 
include_directories(${CMAKE_SOURCE_DIR}/Libs/x86/MNN/include)
find_library(MNN_LIBRARIES MNN ${CMAKE_SOURCE_DIR}/Libs/x86/MNN/lib)

IF(NOT MNN_LIBRARIES)
MESSAGE(FATAL_ERROR "libMNN not found")
ENDIF(NOT MNN_LIBRARIES) 

aux_source_directory(. DIR_SRCS)

add_executable(mnn_demo ${DIR_SRCS})
target_link_libraries(mnn_demo ${MNN_LIBRARIES} jsoncpp_lib ${OpenCV_LIBS} )
set_property(TARGET mnn_demo PROPERTY CXX_STANDARD 11)
