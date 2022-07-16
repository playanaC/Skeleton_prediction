%module Simulator
%include <opencv.i>
%cv_instantiate_all_defaults

%{
    #include "Simulator.hpp" 
    #include <opencv2/core.hpp>
    #include <opencv2/imgproc.hpp>
    #include <opencv2/highgui.hpp>
    #include <iostream>
    #include <string.h>
    #include <fstream>

    #include <dirent.h>
    #include <stdio.h>
    #include <cstdlib>
    #include <math.h>
    #include <cmath>

    #include <algorithm>
    #include <time.h>
    #include <stdlib.h>
    #include <chrono>
    #include <random>
%}

%include "Simulator.hpp"
