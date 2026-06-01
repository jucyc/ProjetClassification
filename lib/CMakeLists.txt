cmake_minimum_required(VERSION 3.10)
project(MLProject)

set(CMAKE_C_STANDARD 99)

# Compiler en bibliothèque partagée
add_library(ml SHARED
    src/linear_model.c
)

# Définir les dossiers d'include
target_include_directories(ml PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Options de compilation
set_target_properties(ml PROPERTIES
    PREFIX ""
    SUFFIX ".so"
)

# Installation
install(TARGETS ml DESTINATION lib)