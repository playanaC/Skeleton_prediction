#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

//#include <sys/time.h>
#include <iostream>
#include <string>
#include <fstream>
//#include <dirent.h>
//#include <stdio.h>
//#include <math.h>
//#include <cmath>


struct Worm
{
    std::string points;
    int total;
    std::vector<float> weights;
    std::vector<float> colors;
};

std::vector<Worm> get_struct_worms(std::string path_data)
{
    char c;
    int cnt_S = 0;
    char init_P[] = "P";
    char init_C[] = "C";
    char init_W[] = "W";
    char finC[] = "\n";
    char pc[] = ";";
    int cnt = 0;
    unsigned int flag_init = 0;
    std::string str_data;

    std::vector<float> weights;
    std::vector<float> colors;

    std::string str_points;
    std::string str_color;
    std::string str_weights;

    std::vector<Worm> Worms;
    int cnt_worms = 0;

    std::ifstream in(path_data);   if (!in) {std::cout << path_data << " not found";   exit(1);}
    while(in.get(c))
    {
        if (flag_init == 1)
        {
            if (c != finC[0])
            {
                if (c == ' '){cnt_S++;}
                str_data += c;
            }
            else
            {
                Worms.push_back(Worm());
                Worms[cnt_worms].points = str_data;
                Worms[cnt_worms].total = cnt_S;
                flag_init=0;
            }
        }
        //****************************************************************************************
        if (flag_init > 1)
        {
            if (c != finC[0])
            {
                if (c != pc[0]){str_data += c;}
                else
                {
                    if (flag_init == 2){colors.push_back(std::atof(str_data.c_str()));}
                    if (flag_init == 3){weights.push_back(std::atof(str_data.c_str()));}
                    str_data = "";
                }                  
            }
            else
            {
                if (flag_init == 2)
                {
                    colors.push_back(std::atof(str_data.c_str()));
                    Worms[cnt_worms].colors = colors;
                }
                if (flag_init == 3)
                {
                    weights.push_back(std::atof(str_data.c_str()));
                    Worms[cnt_worms].weights = weights;
                    cnt_worms++;
                }
                flag_init=0;
            }
        } 
        //****************************************************************************************
        if (flag_init == 0)
        {
            if (c == init_P[0]){flag_init=1;}
            if (c == init_C[0]){flag_init=2;}
            if (c == init_W[0]){flag_init=3;}

            cnt_S=0;
            str_data="";
            colors.clear();
            weights.clear();
        }
    }
    in.close();
    return Worms;
}

std::vector<std::vector<int>> get_data( std::string data, int sz_X )
{
    std::vector<std::vector<int>> vec( sz_X , std::vector<int> (2));
    std::string str_X, str_Y;
    unsigned int flag_init = 0;
    unsigned int cnt_XY = 0;
    unsigned int cnt = 0;
    
    for (int i = 0; i < data.size(); i++)
    {
        cnt=0;
        if (data[i] == ';')
        {
            vec[cnt_XY][1] = std::stoi(str_Y);
            flag_init = 0;
            cnt=1;
            str_Y= "";
            cnt_XY ++;
        }

        if (data[i] == ' ')
        {
            vec[cnt_XY][0] = std::stoi(str_X);
            flag_init = 1;
            cnt=1;
            str_X= "";
        }

        if ((flag_init == 0) && (cnt==0)){ str_X += data[i]; }
        if ((flag_init == 1) && (cnt==0)){ str_Y += data[i]; }
    }
    vec[cnt_XY][1] = std::stoi(str_Y);
    return vec;
}

cv::Mat create_image(std:: string path_data, std::string path_background)
{

    //std:: string path_sim = "C:/Examples_C_CPP/Simulator/swig/New_test/Save_image/ls3_n15_ml11/ls3_n15_ml11dia = 4cond = X pla = 1 living_worms = 15/";
    //std:: string background = "Lifespan13 V2 dia=14 cond=B pla=9";

    //char* cwd = _getcwd( 0, 0 ) ; // **** microsoft specific ****
    //std::string working_directory(cwd) ;
    //std::free(cwd) ;

    //std::string path_background =  working_directory + "/backgrounds/" + background + ".jpg";
    cv::Mat img_background = cv::imread(path_background);

    //std::string path_data = path_sim + "1.pts";
    std::vector<Worm> data_worms = get_struct_worms(path_data);

    int X,Y;
    std::vector<std::vector<int>> XYn;
    std::vector<float> weights;
    std::vector<float> colors;
    cv::Mat BWSk = img_background.clone();
    for(int i=0;i<data_worms.size();i++)
    {
        XYn.clear();
        weights.clear();
        colors.clear();

        XYn = get_data(data_worms[i].points, data_worms[i].total);
        weights = data_worms[i].weights;
        colors = data_worms[i].colors;

        // Graph worm on image
        for (int kl = 0; kl < XYn.size(); kl++)
        {
            X = XYn[kl][0];
            Y = XYn[kl][1];
            cv::circle(BWSk, cv::Point(X,Y),weights[kl], cv::Scalar(colors[kl]),1, 1,0);
        }
    }

    img_background.release();
    data_worms.clear();
    XYn.clear();
    weights.clear();
    colors.clear();
    return BWSk;
}