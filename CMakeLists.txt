cmake_minimum_required(VERSION 3.2)
project(op_sample)
set(CMAKE_CXX_STANDARD 14)

# if libdtu_sdk.so is compiled with ABI=0(default), uncomment Line 6 and comment Line 7
add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)
# if libdtu_sdk.so is compiled with ABI=1, comment Line 6 and uncomment Line 7
#add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=1)

include_directories(/usr/include/dtu /usr/include/dtu/3_0/runtime  ${CMAKE_CURRENT_LIST_DIR}/)
aux_source_directory(${CMAKE_CURRENT_LIST_DIR} DIR_SRCS)
aux_source_directory(${CMAKE_CURRENT_LIST_DIR}/common DIR_SRCS)
link_directories(/usr/lib)
file (GLOB files main.cpp)

set(EXEC "main")
set(OpSupport FALSE)
set(OPfolder "operator") # operator folder name
SET(OPType "NULL" CACHE STRING "select operator sample you want to run")

file(GLOB OPHeaders ${OPfolder}/*.h)

message("Operator support list:")
foreach (OpHeader ${OPHeaders})
    string(REGEX REPLACE ".+/(.+)\\..*" "\\1" OPName ${OpHeader})
    message ("----: ${OPName}")
    message ("      Construct cmd: cmake .. -DOPType=${OPName}")
    if (${OPName} STREQUAL ${OPType})
        set(OpSupport TRUE)
    endif()
endforeach ()

if (OPType STREQUAL NULL) #must specify optype, defult is null
    message(FATAL_ERROR "please specify one sample in folder: ${CMAKE_CURRENT_SOURCE_DIR}/${OPfolder}")
else()
    if (OpSupport STREQUAL FALSE)
        message(WARNING "please make sure ${CMAKE_CURRENT_SOURCE_DIR}/${OPfolder}/${OPType}.h is exits,\
                        or it will build fail!")
    else()
        message("-- ${OPType} sample is constructed")
    endif()
    add_definitions(-DFUNC_FILE=${OPfolder}/${OPType}.h)
    add_executable (${EXEC} ${file} ${DIR_SRCS})
    target_link_libraries(${EXEC} -ldtu_sdk)
endif()