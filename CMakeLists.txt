cmake_minimum_required(VERSION 2.4.7)
PROJECT(OCVVIEWER)

find_package(OpenCV REQUIRED)

include_directories($(OpenCV_INCLUDE_DIR))

aux_source_directory(. SRC_LIST)

foreach(F ${SRC_LIST})
    string(LENGTH ${F} NAMELENGTH)
    string(SUBSTRING ${F} 0 ${NAMELENGTH} FILENAME)
    string(REPLACE .cpp .bin FILENAME1 ${FILENAME})
    add_executable(${FILENAME1} ${F} tinyfiledialogs.c)
    target_link_libraries(${FILENAME1} ${OpenCV_LIBS}  )
endforeach(F)
