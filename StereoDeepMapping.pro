#-------------------------------------------------
#
# Project created by QtCreator 2010-08-12T17:41:41
#
#-------------------------------------------------


TARGET = StereoDeepMapping
CONFIG += console static
CONFIG += release

INCLUDEPATH += "D:\\dev\\opencv\\buildMINGW\\install\\include"
LIBS += -L "D:\\dev\\opencv\\buildMINGW\\install\\lib" -lopencv_core240 -lopencv_highgui240 -lopencv_calib3d240 -lopencv_ml240 -lopencv_video240 -lopencv_imgproc240

#To build a mingw32 version of libdaisy, download this https://github.com/TheFrenchLeaf/libdaisy
#and use cmake
INCLUDEPATH += "D:\\dev\\TheFrenchLeaf-libdaisy-640fbf7\\include"
LIBS += -L "D:\\dev\\TheFrenchLeaf-libdaisy-640fbf7\\lib" -ldaisy

#To build a mingw version of TBB, retrieve src files and do :
#mingw32-make compiler=gcc arch=ia32 runtime=mingw tbb
INCLUDEPATH += "D:\\dev\\tbb40_20120408oss_src\\include"
LIBS += -L "D:\\dev\\tbb40_20120408oss_src\\build\\windows_ia32_gcc_mingw_release" -ltbb

#Add OpenMP
LIBS += -lgomp

TEMPLATE= app

SOURCES += src/main.cpp
