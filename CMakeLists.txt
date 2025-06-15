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
)

# Create library
add_library(mae_lib ${SOURCES})
target_link_libraries(mae_lib "${TORCH_LIBRARIES}" ${OpenCV_LIBS})

# Create training executable
add_executable(train_mae src/train_mae.cpp)
target_link_libraries(train_mae mae_lib "${TORCH_LIBRARIES}" ${OpenCV_LIBS})

# Set RPATH for executables
if (APPLE)
    set_target_properties(train_mae PROPERTIES
        BUILD_RPATH "@loader_path/../lib"
        INSTALL_RPATH "@loader_path/../lib"
    )
elseif (UNIX)
    set_target_properties(train_mae PROPERTIES
        BUILD_RPATH "$ORIGIN/../lib"
        INSTALL_RPATH "$ORIGIN/../lib"
    )
endif()

# Copy libraries to build directory
if (MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET train_mae POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${TORCH_DLLS}
        $<TARGET_FILE_DIR:train_mae>)
endif()