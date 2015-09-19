# StereoDeepMapping
Deep mapping stereo vision based on Dqisy descriptors

Dependencies to install:

*OpenCV

*LibDaisy: https://github.com/etola/libdaisy

*TBB: https://www.threadingbuildingblocks.org/

*MinGW

*QT creator for .pro file (project)


Then customize StereoDeepMapping.pro file with your env

For opencv :

Needed to install the make, gcc and gcc-c++ packages that come with setup.exe from cygwin.

>mkdir build-release && cd build-release
>cmake -DCMAKE_BUILD_TYPE=RELEASE -DWITH_SSE3=ON -DWITH_SSSE3=ON -DWITH_TBB=ON ../
>make install
