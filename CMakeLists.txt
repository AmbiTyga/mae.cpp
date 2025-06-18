cmake_minimum_required(VERSION 3.14)
project(mae_cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

# Set compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Source files
set(SOURCES
    src/mae_model.cpp
    src/data_loader.cpp
    src/mae_inference.cpp
    src/pretrain_utils.cpp
)

# Create library
add_library(mae_lib ${SOURCES})
target_link_libraries(mae_lib "${TORCH_LIBRARIES}" ${OpenCV_LIBS} nlohmann_json::nlohmann_json)
target_include_directories(mae_lib PUBLIC ${CMAKE_SOURCE_DIR}/include)

# Create pretraining executable
add_executable(pretrain_mae src/pretrain_mae.cpp)
target_link_libraries(pretrain_mae 
    mae_lib 
    "${TORCH_LIBRARIES}" 
    ${OpenCV_LIBS}
    nlohmann_json::nlohmann_json
)

# Download and setup cpp-httplib
include(FetchContent)
FetchContent_Declare(
    httplib
    GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
    GIT_TAG v0.14.1
)
FetchContent_MakeAvailable(httplib)

# Download and setup nlohmann/json
FetchContent_Declare(
    json
    URL https://github.com/nlohmann/json/releases/download/v3.11.2/json.tar.xz
)
FetchContent_MakeAvailable(json)

# Download and setup cxxopts
FetchContent_Declare(
    cxxopts
    GIT_REPOSITORY https://github.com/jarro2783/cxxopts.git
    GIT_TAG v3.1.1
)
FetchContent_MakeAvailable(cxxopts)

# Create inference server executable (for TorchScript models)
add_executable(mae_server src/mae_server.cpp)
target_link_libraries(mae_server 
    mae_lib 
    "${TORCH_LIBRARIES}" 
    ${OpenCV_LIBS}
    httplib::httplib
    nlohmann_json::nlohmann_json
    cxxopts::cxxopts
)

# Create native model server executable
add_executable(mae_native_server src/mae_native_server.cpp)
target_link_libraries(mae_native_server 
    mae_lib 
    "${TORCH_LIBRARIES}" 
    ${OpenCV_LIBS}
    httplib::httplib
    nlohmann_json::nlohmann_json
    cxxopts::cxxopts
)


# Set RPATH for executables
if (APPLE)
    set_target_properties(pretrain_mae mae_server mae_native_server PROPERTIES
        BUILD_RPATH "@loader_path/../lib"
        INSTALL_RPATH "@loader_path/../lib"
    )
elseif (UNIX)
    set_target_properties(pretrain_mae mae_server mae_native_server PROPERTIES
        BUILD_RPATH "$ORIGIN/../lib"
        INSTALL_RPATH "$ORIGIN/../lib"
    )
endif()

# Copy libraries to build directory
if (MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET pretrain_mae POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${TORCH_DLLS}
        $<TARGET_FILE_DIR:pretrain_mae>)
endif()