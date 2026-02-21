cmake_minimum_required(VERSION 3.14)
project(drums_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -ffast-math -march=native")

# ── Dependencies ──────────────────────────────────────────────────────────────

include(FetchContent)

# 1. Eigen (Math)
FetchContent_Declare(
  Eigen3
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG        3.4.0
)
FetchContent_MakeAvailable(Eigen3)

# 2. Spectra (Eigenvalue Solver)
FetchContent_Declare(
  Spectra
  GIT_REPOSITORY https://github.com/yixuan/spectra.git
  GIT_TAG        v1.0.1
)
FetchContent_MakeAvailable(Spectra)

# 3. SDL2 (Window/Audio)
FetchContent_Declare(
  SDL2
  GIT_REPOSITORY https://github.com/libsdl-org/SDL.git
  GIT_TAG        release-2.30.0
)
FetchContent_MakeAvailable(SDL2)

# 4. Dear ImGui
FetchContent_Declare(
  imgui
  GIT_REPOSITORY https://github.com/ocornut/imgui.git
  GIT_TAG        v1.90.4
)
FetchContent_MakeAvailable(imgui)

# 5. ImPlot
FetchContent_Declare(
  implot
  GIT_REPOSITORY https://github.com/epezent/implot.git
  GIT_TAG        v0.16
)
FetchContent_MakeAvailable(implot)

# 6. CDT (Constrained Delaunay Triangulation - header-only)
FetchContent_Declare(
  CDT
  GIT_REPOSITORY https://github.com/artem-ogre/CDT.git
  GIT_TAG        1.4.1
)
FetchContent_MakeAvailable(CDT)

# ── Sources ───────────────────────────────────────────────────────────────────

# Create a library for ImGui so we don't recompile it every time
add_library(imgui_lib STATIC
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_sdl2.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_sdlrenderer2.cpp
)
target_include_directories(imgui_lib PUBLIC 
    ${imgui_SOURCE_DIR} 
    ${imgui_SOURCE_DIR}/backends
    ${SDL2_SOURCE_DIR}/include # For SDL2 backend
)
target_link_libraries(imgui_lib PUBLIC SDL2::SDL2)

# ImPlot library
add_library(implot_lib STATIC
    ${implot_SOURCE_DIR}/implot.cpp
    ${implot_SOURCE_DIR}/implot_items.cpp
)
target_include_directories(implot_lib PUBLIC ${implot_SOURCE_DIR})
target_link_libraries(implot_lib PUBLIC imgui_lib)

# Main Executable
add_executable(drums_cpp
    src/main.cpp
)

target_link_libraries(drums_cpp PRIVATE
    Eigen3::Eigen
    SDL2::SDL2
    SDL2::SDL2main
    imgui_lib
    implot_lib
)

# Spectra is header-only but needs to be in include path
target_include_directories(drums_cpp PRIVATE ${spectra_SOURCE_DIR}/include ${cdt_SOURCE_DIR}/CDT/include)

# Test Executable
add_executable(test_polygon
    src/test_polygon.cpp
)

target_link_libraries(test_polygon PRIVATE
    Eigen3::Eigen
)

target_include_directories(test_polygon PRIVATE ${spectra_SOURCE_DIR}/include ${cdt_SOURCE_DIR}/CDT/include)
