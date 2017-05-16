cmake_minimum_required(VERSION 3.6)
project(ocr_neural_net)


set(COMPILE_FLAGS "-std=c++1y -Wall -Wextra -pedantic -Ofast")
file(GLOB SOURCES "src/*.cpp" "src/*.h")

add_definitions(${COMPILE_FLAGS})
add_executable(${PROJECT_NAME} ${SOURCES})
# target_link_libraries(${PROJECT_NAME} <library_name>)