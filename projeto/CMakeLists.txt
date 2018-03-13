
cmake_minimum_required(VERSION 2.8)

project(projeto)

find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

add_executable(projeto projeto.cxx)

target_link_libraries(projeto ${ITK_LIBRARIES})


