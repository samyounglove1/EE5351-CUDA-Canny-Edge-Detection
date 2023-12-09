 # Parallel Canny Edge Detection


 ## Installation

 Note that these instructions are for compiling and running the application using a Linux environment. In order to run all demo modes you will also need an installation of FFmpeg, instructions to set that up can be found [here.](https://trac.ffmpeg.org/wiki/CompilationGuide/Ubuntu)

Additionally you will need working install of [OpenCV](https://docs.opencv.org/4.x/d0/d3d/tutorial_general_install.html). Ensure to build the library with the flags .
- `-DOPENCV_GENERATE_PKGCONFIG=ON`
- `-DWITH_FFMPEG=ON`

Finally you'll need to have x11 forwarding set up.

## Compiling and Running

To compile the application, simply 'cd' into the build directory and run the make file. The makefile is setup to compile for a RTX 20 series GPU however the argument `SM` can be used to explicitly define the compute/SM version for your graphics card. You can figure out which version you need [here.](https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/)

When running the application there are a few modes of operation. You can supply a `-m` argument to specify a mode of operation for demo selection. Currently there are 2 modes, multi-image benchmarking and a live video demo, if no arguments are specified then the app will default to the first of these 2 modes. Additionally you can supply a `-i` that can be followed by an image path, this will run the canny edge detection on the specified image and show the result. 
