cmake_minimum_required(VERSION 3.10)
project(main)

find_package(OpenCV REQUIRED)

add_executable(main main.cpp utils/MultipleImageWindow.cpp)

target_include_directories(main PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/utils
)

target_link_libraries(main ${OpenCV_LIBS})
