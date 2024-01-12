cmake_minimum_required(VERSION 3.26)
project(chatllm)

set(CMAKE_CXX_STANDARD 20)

add_executable(chatllm main.cpp
        chatllm.cpp
        chatllm.h
        main.cpp
        chatllm.cpp
        chatllm.h
)
add_subdirectory(third-party/sentencepiece)
target_link_libraries(chatllm PUBLIC sentencepiece)

include_directories(third_party/ggml/include/ggml third_party/ggml/src)
add_subdirectory(third-party/ggml)
target_link_libraries(chatllm PUBLIC ggml)
