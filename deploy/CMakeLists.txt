cmake_minimum_required(VERSION 3.0)

project(deploy)

find_package(Torch REQUIRED PATHS /home/zhengxuping/Softwares/libtorch/share/cmake/Torch)

include_directories(/usr/include)
include_directories(/usr/local/include)
include_directories(/home/zhengxuping/Softwares/libtorch/include)

link_directories(/usr/lib)
link_directories(/usr/local/lb)
link_directories(/home/zhengxuping/Softwares/libtorch/lib)

link_libraries(${TORCH_LIBRARIES})

add_definitions(-O3 -g -std=c++11)

add_executable(deploy src/Deploy.cpp)