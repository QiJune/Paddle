set(OUTPUT_DIR
    "${CMAKE_CURRENT_BINARY_DIR}/build")

file(GLOB_RECURSE V2_PY_FILES ./paddle/framework *.py)


set(PY_FILES paddle/__init__.py
  ${V2_PY_FILES})

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/setup.py)


add_custom_command(OUTPUT ${PROJ_ROOT}/python/paddle/framework/core.so
        COMMAND cmake -E copy $<TARGET_FILE:paddle_pybind> ${PROJ_ROOT}/python/paddle/framework/core.so
        DEPENDS paddle_pybind)
add_custom_target(copy_paddle_pybind ALL DEPENDS ${PROJ_ROOT}/python/paddle/framework/core.so)


add_custom_command(OUTPUT ${OUTPUT_DIR}/.timestamp
    COMMAND env ${py_env} ${PYTHON_EXECUTABLE} setup.py bdist_wheel
    COMMAND ${CMAKE_COMMAND} -E touch ${OUTPUT_DIR}/.timestamp
    DEPENDS copy_paddle_pybind framework_py_proto ${PY_FILES} ${external_project_dependencies})

add_custom_target(paddle_python ALL DEPENDS
    ${OUTPUT_DIR}/.timestamp)

set(PADDLE_PYTHON_PACKAGE_DIR ${CMAKE_CURRENT_BINARY_DIR}/dist/)

if (WITH_TESTING)
  add_subdirectory(paddle/framework/tests)
endif()

install(DIRECTORY ${PADDLE_PYTHON_PACKAGE_DIR}
    DESTINATION opt/paddle/share/wheels
)
