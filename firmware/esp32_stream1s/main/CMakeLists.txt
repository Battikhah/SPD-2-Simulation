if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/generated/model_data.cc")
    message(FATAL_ERROR "Run scripts/gen_esp32_example.py to generate main/generated before building")
endif()
idf_component_register(SRCS "main.cc" "../../logmel/logmel.c" "generated/model_data.cc"
                       INCLUDE_DIRS "." "../../logmel" "generated"
                       REQUIRES esp_timer heap esp_system)
# Single-precision log matched host parity exactly (MSVC); re-check with the board's newlib logf.
target_compile_definitions(${COMPONENT_LIB} PRIVATE LOGMEL_USE_LOGF)
