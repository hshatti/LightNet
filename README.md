# <p align="center">LightNet</p>
Implemented in pure pascal LightNet is an artificial intelligence neural network library Inspired by 
[Darknet](https://pjreddie.com/darknet) and [YOLOv7 github](https://github.com/AlexeyAB/darknet) C library, 
this implementation can run most of the darknet including YOLO models nativly and self dependently no external libraries required.

- Currently training & inference runs on CPU , however i have notice that inference runs faster than the Optimized C CPU implementation
- Optinally, using `-dOPENBLAS`  or `-dMKL` ( Custom options in Lazarus-Ide for example ) will enable "libopenblas" or "Intel's Math Kernel Library" which is further optimized for CPU, however, on lazarus no significant improvement was noticed.
- Before running the test make sure that [yolov7.weights](https://sourceforge.net/projects/darknet-yolo.mirror/files/yolov4/) file is placed next to the test project executable.
- Other .weights models should work (check "cfg" folder) 
- More Examples coming soon


### Work pending 
- OpenCL aand CUDA implementation
- Test on other platforms (MacOS, Linux)
- Test on Single Board Computers (NVIDIA Jetson, Raspberry PI, etc..)


tests are provided for both Lazarus and Delphi