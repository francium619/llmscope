cmake_minimum_required(VERSION 3.28)
project(llmscope LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(LLMSCOPE_WITH_LLAMA "Build the llama.cpp tracer backend" ON)
option(LLMSCOPE_BUILD_TESTS "Build unit tests" ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# ---------------------------------------------------------------------------
# Dependencies (vendored under external/, built from source)
# ---------------------------------------------------------------------------

set(FTXUI_BUILD_DOCS     OFF CACHE BOOL "" FORCE)
set(FTXUI_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(FTXUI_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(FTXUI_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
set(FTXUI_QUIET          ON  CACHE BOOL "" FORCE)
add_subdirectory(external/FTXUI EXCLUDE_FROM_ALL)

if(LLMSCOPE_WITH_LLAMA)
  # We consume llama.cpp entirely as a library: no examples, no tools, no
  # server, and no curl dependency. The tracer hooks it through the public
  # cb_eval callback, so nothing in its source tree is patched.
  set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
  set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(LLAMA_BUILD_TOOLS    OFF CACHE BOOL "" FORCE)
  set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "" FORCE)
  set(LLAMA_BUILD_COMMON   OFF CACHE BOOL "" FORCE)
  set(LLAMA_CURL           OFF CACHE BOOL "" FORCE)
  set(BUILD_SHARED_LIBS    OFF CACHE BOOL "" FORCE)
  add_subdirectory(external/llama.cpp EXCLUDE_FROM_ALL)
endif()

# ---------------------------------------------------------------------------
# trace_core - records, ring buffer, topology, trace files, sources.
# Deliberately depends on nothing: no llama.cpp, no FTXUI. This is what lets the
# TUI be built and tested without a model and keeps the format portable.
# ---------------------------------------------------------------------------

add_library(llmscope_core STATIC
  src/trace_core/name_table.cpp
  src/trace_core/topology.cpp
  src/trace_core/trace_file.cpp
  src/trace_core/live_source.cpp
  src/trace_core/fake_source.cpp
)
target_include_directories(llmscope_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_features(llmscope_core PUBLIC cxx_std_17)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(llmscope_core PRIVATE -Wall -Wextra)
endif()

find_package(Threads REQUIRED)
target_link_libraries(llmscope_core PUBLIC Threads::Threads)

# ---------------------------------------------------------------------------
# tracer - the llama.cpp hook. The only module that sees ggml headers.
# ---------------------------------------------------------------------------

if(LLMSCOPE_WITH_LLAMA)
  add_library(llmscope_tracer STATIC
    src/tracer/tracer.cpp
    src/tracer/tensor_stats.cpp
  )
  target_include_directories(llmscope_tracer PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_link_libraries(llmscope_tracer PUBLIC llmscope_core llama ggml)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(llmscope_tracer PRIVATE -Wall -Wextra)
  endif()
endif()

# ---------------------------------------------------------------------------
# tui - FTXUI front end. Knows nothing about llama.cpp.
# ---------------------------------------------------------------------------

add_library(llmscope_tui STATIC
  src/tui/ui_state.cpp
  src/tui/app.cpp
)
target_include_directories(llmscope_tui PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(llmscope_tui PUBLIC llmscope_core ftxui::screen ftxui::dom ftxui::component)

# ---------------------------------------------------------------------------
# executable
# ---------------------------------------------------------------------------

add_executable(llmscope src/app/main.cpp)
target_link_libraries(llmscope PRIVATE llmscope_core llmscope_tui)

if(LLMSCOPE_WITH_LLAMA)
  target_link_libraries(llmscope PRIVATE llmscope_tracer)
  target_compile_definitions(llmscope PRIVATE LLMSCOPE_WITH_LLAMA=1)
endif()

# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

if(LLMSCOPE_BUILD_TESTS)
  enable_testing()
  add_executable(llmscope_tests tests/test_trace_core.cpp)
  target_link_libraries(llmscope_tests PRIVATE llmscope_core)
  add_test(NAME trace_core COMMAND llmscope_tests)
endif()
