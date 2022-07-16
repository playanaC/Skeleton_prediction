%module Create_image
%include <opencv.i>
%cv_instantiate_all_defaults

%{
    #include "Create_image.hpp" 
    #include <opencv2/core.hpp>
    #include <opencv2/imgproc.hpp>
    #include <opencv2/highgui.hpp>
    #include <iostream>
    #include <string.h>
    #include <fstream>
%}

%include "Create_image.hpp"
