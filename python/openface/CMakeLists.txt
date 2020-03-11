# Copyright
# Joao Avelino 2019
# All rights reserved

set(PYTHON_FILES
	pyopenface.py
	__init__.py
	openface_python.cpp)

include_directories(${LandmarkDetector_SOURCE_DIR}/include)

pybind11_add_module(pyopenface openface_python.cpp util/conversions.cpp)

target_link_libraries(pyopenface PRIVATE pybind11::module dlib LandmarkDetector FaceAnalyser GazeAnalyser Utilities)

set_target_properties(pyopenface PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}"
    SUFFIX "${PYTHON_MODULE_EXTENSION}") 

configure_file(__init__.py __init__.py)

install(TARGETS pyopenface DESTINATION python)
install(DIRECTORY ${CMAKE_CURRENT_DIRECTORY}/ DESTINATION python/openface
    FILES_MATCHING PATTERN "*.so")
install(DIRECTORY ${CMAKE_CURRENT_DIRECTORY}/ DESTINATION python/openface
    FILES_MATCHING PATTERN "*.py")
