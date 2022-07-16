#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <sys/time.h>
#include <iostream>
#include <fstream>
#include <string>
#include <dirent.h>
#include <stdio.h>
#include <cstdlib>
#include <math.h>
#include <cmath>

#include <stdlib.h> 
#include <time.h>
#include <algorithm>

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

float pi = 3.14159265359;

#include <chrono>
#include <random>

struct Worm
{
    std:: string name_xml;
    std::vector<std::string> points;
    std::vector<int> XYI;
    std::vector<int> total;
    std::vector<float> weights;
    std::vector<float> colors;
    std::vector<float> MA;
    int Radio;
    int Angle;
    int Rx;
    int Ry;
    int Velocity;
};

std::vector<std::string> read_trackInfo(std::string path_data_track, std::vector<float> &media_colors, std::vector<int> &tam_velocity, std::vector<int> &r_xml)
{
    char c;
    int cnt = 0;
    char initC[] = ",";
    char finC[] = "\n";
    std::string str_data;
    std::vector<std::string> xml_names;
    
    std::ifstream in( path_data_track );   if ( !in ) { std::cout << path_data_track << " not found";   exit( 1 ); }
    while( in.get( c ) )
    {
        if (c == finC[0])
        {
            r_xml.push_back(std::stoi(str_data));
            cnt=-1;
        }

        if ((c != initC[0]) && (c != finC[0])){str_data += c;}
        else
        {
            if (cnt == 0){xml_names.push_back(str_data);}
            if (cnt == 1){media_colors.push_back(std::atof(str_data.c_str()));}
            if (cnt == 2){tam_velocity.push_back(std::stoi(str_data));}
            
            str_data = "";
            cnt++;
        }
    }
    in.close();
    return xml_names;
}

std::vector<std::string> read_backgrounds_Info(std::string path_data_track, std::vector<int> &Brc, std::vector<int> &Brx, std::vector<int> &Bry)
{
    char c;
    int cnt = 0;
    char initC[] = ",";
    char finC[] = "\n";
    std::string str_data;
    std::vector<std::string> backgrounds_names;
    
    std::ifstream in( path_data_track );   if ( !in ) { std::cout << path_data_track << " not found";   exit( 1 ); }
    while( in.get( c ) )
    {
        if (c == finC[0])
        {
            Bry.push_back(std::stoi(str_data));
            cnt=-1;
        }

        if ((c != initC[0]) && (c != finC[0])){str_data += c;}
        else
        {
            if (cnt == 0){backgrounds_names.push_back(str_data);}
            if (cnt == 1){Brc.push_back(std::stoi(str_data));}
            if (cnt == 2){Brx.push_back(std::stoi(str_data));}
            
            str_data = "";
            cnt++;
        }
    }
    in.close();
    return backgrounds_names;
}

std::vector<std::string> xml_data( std::string filename, std::vector<int> &total, std::vector<float> &weight, std::vector<float> &color)
{
    char c;
    int cnt_S = 0;
    char initC[] = "<pose>[";
    char init_Color[] = "<Colors>[";
    char init_Weight[] = "<Weight>[";
    char finC[] = "]";
    char finCWC[] = ";";
    unsigned int cnt = 0;
    unsigned int cnt_color = 0;
    unsigned int cnt_Weight = 0;
    unsigned int flag_init = 0;
    unsigned int flag_var = 0;

    std::string str_init;
    std::string str_init_Color;
    std::string str_init_Weight;

    std::string S_initC = initC;
    std::string S_init_Color = init_Color;
    std::string S_init_Weight = init_Weight;

    std::vector<std::string> collection;
    std::string str_points;
    std::string str_weights_color;
    
    std::ifstream in( filename );   if ( !in ) { std::cout << filename << " not found";   exit( 1 ); }
    while( in.get( c ) )
    {
        if (flag_init == 1)
        {
            if (c == ' '){cnt_S++;}
            if (finC[cnt]== c) {cnt++;}
            else {cnt=0;}

            if ((cnt == strlen(finC)) && (flag_var==1))
            {
                flag_init = 0;
                cnt=0;
                collection.push_back(str_points);
                total.push_back(cnt_S);
            }
            //******************************************************
            if (cnt != strlen(finC))
            {
                if ((finCWC[0] != c) && (flag_var>1))
                {
                    str_weights_color+= c;
                }
                else
                {
                    if (flag_var==2){weight.push_back(std::atof(str_weights_color.c_str()));}
                    if (flag_var==3){color.push_back(std::atof(str_weights_color.c_str()));}
                    str_weights_color="";
                }
            }
            else{
                    if (flag_var==2){weight.push_back(std::atof(str_weights_color.c_str()));}
                    if (flag_var==3){color.push_back(std::atof(str_weights_color.c_str()));}
                    flag_init = 0;
                    str_weights_color="";
                }
            str_points += c;
        }

        if (flag_init == 0)
        {
            //******************************************************
            if (initC[cnt]== c){str_init += c; cnt++;}
            else {str_init = "";cnt=0;}
            //******************************************************
            if (init_Color[cnt_color]== c){str_init_Color += c; cnt_color++;}
            else {str_init_Color = "";cnt_color=0;}
            //******************************************************
            if (init_Weight[cnt_Weight]== c){str_init_Weight += c; cnt_Weight++;}
            else {str_init_Weight = "";cnt_Weight=0;}
            //******************************************************
            if (str_init.compare(S_initC)==0)
            {
                str_points="";
                flag_init = 1;
                flag_var = 1;
                cnt=0;
                cnt_S=0;
            }
            //******************************************************
            if (str_init_Weight.compare(S_init_Weight)==0)
            {
                str_init_Weight="";
                str_points="";
                flag_init = 1;
                flag_var = 2;
                cnt_Weight = 0;
            }
            //******************************************************
            if (str_init_Color.compare(S_init_Color)==0)
            {
                str_init_Color="";
                str_points="";
                flag_init = 1;
                flag_var = 3;
                cnt_color = 0;
            }
        }
    }
    in.close();
    return collection;
}

std::vector<std::vector<int>> worm_aggregation(int worms, int parall, int coll)
{
    int parall_ = (int)floor(((float)parall)/((float)2));
    int coll_ = (int)floor(((float)coll)/((float)2));
    //std::vector<int> vec_worms;
    std::vector<std::vector<int>> vec_worms( worms , std::vector<int> (2));

    int cnt = 0;
    while (cnt<worms)
    {
        if (parall_ !=0)
        {
            //vec_worms.push_back(0);
            //vec_worms.push_back(3);
            vec_worms[cnt][0] = 0;
            vec_worms[cnt][1] = 0;
            vec_worms[cnt+1][0] = 3;
            vec_worms[cnt+1][1] = 0;
            cnt = cnt+2;
            parall_ = parall_ -1;
        }
        else
        {
            if (coll_ !=0)
            {
                //vec_worms.push_back(0);
                //vec_worms.push_back(2);
                vec_worms[cnt][0] = 0;
                vec_worms[cnt][1] = 0;
                vec_worms[cnt+1][0] = 2;
                vec_worms[cnt+1][1] = 0;
                cnt = cnt+2;
                coll_ = coll_ -1;
            }
            else
            {
                //vec_worms.push_back(1);
                vec_worms[cnt][0] = 1;
                vec_worms[cnt][1] = 0;
                cnt = cnt+1;
            }
        }
    }
    return vec_worms;
}

int worm_rotate(int rxml, int R, int CC1, int CC2, int &R_random, std::vector<float> &MA, int &rx, int &ry)
{
    std::mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count());
    std::uniform_int_distribution<int> A_random(0,359);
    std::uniform_int_distribution<int> Rdist(0,R - (rxml * 2) - 1);
    
    int A_random_int;
    float sa, ca;
//     int Rdist = R - (rxml * 2) - 1;
//     int A_random = rand()%(360);
    A_random_int = A_random(rng);

//     R_random = rand()%(Rdist);
    R_random = Rdist(rng);
    sa = sin((float)A_random_int*(pi/180));
    ca = cos((float)A_random_int*(pi/180));

    MA.push_back(ca);
    MA.push_back(sa);
    MA.push_back(-sa);
    MA.push_back(ca);

    rx = round((float)R_random * sa) + CC1;
    ry = round((float)R_random * ca) + CC2;
    
    return A_random_int;
}

std::vector<std::vector<int>> get_data( std::string data, int sz_X )
{
    std::vector<std::vector<int>> vec( sz_X , std::vector<int> (2));
    std::string str_X, str_Y;
    unsigned int flag_init = 0;
    unsigned int cnt_XY = 0;
    unsigned int cnt = 0;
    int i;
    
    for (i = 0; i < data.size(); i++)
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
void scale_data(std::vector<float> weights, std::vector<float> color, int l, std::vector<float> &weights_scale, std::vector<float> &colors_scale)
{
    int n, j;
    float temp,m, temp_weight;

    n = weights.size();
    m = (float)n/((float)l+1);

    for(j=1;j<l+1;j++)
    {
        temp = ((float)j*m) - (m/2);
        temp_weight = (weights[round(temp - 1) - 1]+weights[round(temp)- 1]+weights[round(temp + 1) - 1])/9;
        if (temp_weight<1){weights_scale.push_back(1);}
        else {weights_scale.push_back(temp_weight);}
        colors_scale.push_back((color[round(temp - 1) - 1]+color[round(temp)- 1]+color[round(temp + 1) - 1])/3);
    }
}

int dirExists(const char * path)
{
    struct stat info;
    //const char * path = path_string.c_str();

    if(stat( path, &info ) == 0)
        return 1;
    else
        return 0;
}

std::vector<Worm> get_data_worms(std::vector<std::vector<int>> &vec_worms, int total_image, int velocity, int worms, std:: string path_tracks, std::vector<std::string> xml_names, std::vector<float> media_colors, std::vector<int> tam_velocity, std::vector<int> r_xml, int R, int CC1, int CC2, int parall, int coll)
{
    int i, j, rxml, Rdist, R_random, R_random_, A_random, rx, ry, rx_, ry_;
    int min_sz, max_sz, random_sz, radio;
    int cnt_worm = 0;
    float X_ran, Y_ran, sa, ca;
    std::string nameXmls, path_track_xml;
    std::vector<Worm> Worms_struct;
    std::vector<int> index_media, index_tam, idx_tam;

    std::vector<int> XYI;
    std::vector<int> total;
    std::vector<std::string> points;

    std::vector<int> total_;
    std::vector<std::string> points_;
    
    std::vector<int> total_temp;
    std::vector<std::string> points_temp;
    
    std::vector<int> total_inv;
    std::vector<std::string> points_inv;

    std::vector<std::vector<int>> XYn;
    std::vector<std::vector<int>> XYn_;
    std::vector<float> weights_;

    std::vector<float> weights;
    std::vector<float> colors;
    std::vector<float> MA;

    for(i=0; i<media_colors.size(); i++)
    {
        if (media_colors[i]<25)
        {
            index_media.push_back(i); //Find files that satisfy color restictions
            if (tam_velocity[i]>=(total_image*velocity)){index_tam.push_back(i);}//Find files that satisfy color restictions
        } 
    }
    
//     srand(time(0));
//     std::srand(unsigned(std::time(0)));
//     std::random_device rd;
//     std::mt19937 g(rd());
    std::mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count());
//     std::uniform_int_distribution<int> distribution(0,359);

    //random generator (random_shuffle)
    for (i=1; i<index_tam.size(); ++i) {idx_tam.push_back(i);}
//     std::random_shuffle (idx_tam.begin(), idx_tam.end());
    std::shuffle(idx_tam.begin(), idx_tam.end(), rng);

    vec_worms = worm_aggregation(worms, parall, coll);
    for (i=0; i<worms; ++i)
    {
        nameXmls="";
        path_track_xml = "";
        points.clear();
        points_.clear();
        total.clear();
        total_.clear();
        XYn.clear();
        XYn_.clear();

        weights.clear();
        colors.clear();
        
        nameXmls = xml_names[index_tam[idx_tam[i]]];
        path_track_xml = path_tracks + nameXmls;
        points = xml_data(path_track_xml,total,weights,colors);
        rxml = r_xml[index_tam[idx_tam[i]]];

        Worms_struct.push_back(Worm());
        Worms_struct[cnt_worm].name_xml = nameXmls;
        Worms_struct[cnt_worm].points = points;
        Worms_struct[cnt_worm].total = total;
        Worms_struct[cnt_worm].weights = weights;
        Worms_struct[cnt_worm].colors = colors;
        Worms_struct[cnt_worm].Velocity = velocity;
        
        if (vec_worms[i][0]==3)
        {
            total_temp.clear();
            points_temp.clear();
            weights_.clear();
            points_ = Worms_struct[cnt_worm-1].points;
            total_ = Worms_struct[cnt_worm-1].total;
            random_sz = rand()%(100);

            if (points_.size() > points.size())
            {
                weights_ = Worms_struct[cnt_worm-1].weights;
                radio = *std::max_element(std::begin(weights_), std::end(weights_));
                total_temp = total_;
                points_temp = points_;
                if (random_sz>50)
                {
                    points_inv.clear();
                    total_inv.clear();
                    for (j=0;j<points_.size();j++)
                    {
                        points_inv.push_back(points_[points_.size()-1-j]);
                        total_inv.push_back(total_[total_.size()-1-j]);
                    }
                    total_.clear();
                    points_.clear();
                    total_ = total_inv;
                    points_ = points_inv;
                    vec_worms[i][1] = 4;
                }
            }
            else
            {
                total_.clear();
                points_.clear();
                radio = *std::max_element(std::begin(weights), std::end(weights));
                total_temp = total;
                points_temp = points;

                if (random_sz>50)
                {
                    points_inv.clear();
                    total_inv.clear();
                    for (j=0;j<points.size();j++)
                    {
                        points_inv.push_back(points[points.size()-1-j]);
                        total_inv.push_back(total[total.size()-1-j]);
                    }
                    total_ = total_inv;
                    points_ = points_inv;
                    vec_worms[i][1] = 4;
                }
                else
                {
                    total_ = total;
                    points_ = points;
                }
            }

            XYI.clear();
            XYn_ = get_data(points_[0],total_[0]);
            XYI.push_back(XYn_[0][0]);
            XYI.push_back(XYn_[0][1]);
            Worms_struct[cnt_worm-1].XYI = XYI;
            Worms_struct[cnt_worm-1].points = points_;
            Worms_struct[cnt_worm-1].total = total_;

            XYI.clear();
            XYn = get_data(points_temp[0],total_temp[0]);
            XYI.push_back(XYn[0][0]);
            XYI.push_back(XYn[0][1]);
            Worms_struct[cnt_worm].XYI = XYI;
            Worms_struct[cnt_worm].points = points_temp;
            Worms_struct[cnt_worm].total = total_temp;

            MA.clear();
            A_random = worm_rotate(rxml, R, CC1, CC2, R_random_, MA, rx_, ry_);
            Worms_struct[cnt_worm-1].Radio = R_random_;
            Worms_struct[cnt_worm-1].Angle = A_random;
            Worms_struct[cnt_worm-1].MA = MA;
            Worms_struct[cnt_worm-1].Rx = rx_;
            Worms_struct[cnt_worm-1].Ry = ry_;

            sa = sin((float)A_random*(pi/180));
            ca = cos((float)A_random*(pi/180));
            random_sz = rand()%(100);
            if (random_sz>50)
            {
                X_ran = ((float)(R_random_ - radio)) * sa;
                Y_ran = ((float)(R_random_ - radio)) * ca;
            }
            else
            {
                X_ran = ((float)(R_random_ + radio)) * sa;
                Y_ran = ((float)(R_random_ + radio)) * ca;
            }

            rx = (int)(round(X_ran)) + CC1;
            ry = (int)(round(Y_ran)) + CC2;

            Worms_struct[cnt_worm].Radio = R_random_;
            Worms_struct[cnt_worm].Angle = A_random;
            Worms_struct[cnt_worm].MA = MA;
            Worms_struct[cnt_worm].Rx = rx;
            Worms_struct[cnt_worm].Ry = ry;
        }

        if (vec_worms[i][0]==2)
        {
            points_ = Worms_struct[cnt_worm-1].points;
            total_ = Worms_struct[cnt_worm-1].total;

            if (points_.size() < points.size()){min_sz=points_.size();}
            else{min_sz=points.size();}

            random_sz = rand()%(min_sz);
            random_sz = int(floor((float)random_sz/(float)velocity));
            if (random_sz==0){random_sz=3;}

            XYn_ = get_data(points_[random_sz],total_[random_sz]);
            XYn = get_data(points[random_sz],total[random_sz]);

            XYI.clear();
            random_sz = rand()%(XYn_.size());
            XYI.push_back(XYn_[random_sz][0]);
            XYI.push_back(XYn_[random_sz][1]);
            Worms_struct[cnt_worm-1].XYI = XYI;

            XYI.clear();
            random_sz = rand()%(XYn.size());
            XYI.push_back(XYn[random_sz][0]);
            XYI.push_back(XYn[random_sz][1]);
            Worms_struct[cnt_worm].XYI = XYI;

            MA.clear();
            A_random = worm_rotate(rxml, R, CC1, CC2, R_random_, MA, rx_, ry_);
            Worms_struct[cnt_worm-1].Radio = R_random_;
            Worms_struct[cnt_worm-1].Angle = A_random;
            Worms_struct[cnt_worm-1].MA = MA;
            Worms_struct[cnt_worm-1].Rx = rx_;
            Worms_struct[cnt_worm-1].Ry = ry_;

            MA.clear();
            A_random = worm_rotate(rxml, R, CC1, CC2, R_random, MA, rx, ry);
            Worms_struct[cnt_worm].Radio = R_random_;
            Worms_struct[cnt_worm].Angle = A_random;
            Worms_struct[cnt_worm].MA = MA;
            Worms_struct[cnt_worm].Rx = rx_;
            Worms_struct[cnt_worm].Ry = ry_; 
        }

        if (vec_worms[i][0]==1)
        {
            XYI.clear();
            XYn = get_data(points[0],total[0]);
            XYI.push_back(XYn[0][0]);
            XYI.push_back(XYn[0][1]);
            Worms_struct[cnt_worm].XYI = XYI;

            MA.clear();
            A_random = worm_rotate(rxml, R, CC1, CC2, R_random, MA, rx, ry);
            Worms_struct[cnt_worm].Radio = R_random;
            Worms_struct[cnt_worm].Angle = A_random;
            Worms_struct[cnt_worm].MA = MA;
            Worms_struct[cnt_worm].Rx = rx;
            Worms_struct[cnt_worm].Ry = ry;
        }
       cnt_worm ++; 
    }
    return Worms_struct;
}

float angXY(float ang, int x1, int y1, int x2, int y2, int &x_temp, int &y_temp)
{
    float new_d, xr, yr;

    if ((ang>=0) && (ang<45))
    {
        x_temp = x1 + 1;
        y_temp = y1;
    }

    if ((ang>=45) && (ang<90))
    {
        x_temp = x1 + 1;
        y_temp = y1 + 1;
    }

    if ((ang>=90) && (ang<135))
    {
        x_temp = x1;
        y_temp = y1 + 1;
    }

    if ((ang>=135) && (ang<180))
    {
        x_temp = x1 - 1;
        y_temp = y1 + 1;
    }

    if ((ang>=180) && (ang<225))
    {
        x_temp = x1 - 1;
        y_temp = y1;
    }

    if ((ang>=225) && (ang<270))
    {
        x_temp = x1 - 1;
        y_temp = y1 - 1;
    }

    if ((ang>=270) && (ang<315))
    {
        x_temp = x1;
        y_temp = y1 - 1;
    }

    if ((ang>=315) && (ang<=360))
    {
        x_temp = x1 + 1;
        y_temp = y1;
    }

    xr = (float)(x2-x_temp);
    yr = (float)(y2-y_temp);

    new_d = sqrt(pow(xr, 2) + pow(yr, 2));
    return new_d;
}

void check_XY(std::vector<int> X, std::vector<int> Y, std::vector<int> &XF, std::vector<int> &YF)
{
    int kl, x1, y1, x2, y2;
    int x_temp, y_temp;
    float d, xr, yr, ang;

    for (kl = 0; kl < X.size()-1; kl++)
    {
        x1 = X[kl];
        y1 = Y[kl];
        x2 = X[kl+1];
        y2 = Y[kl+1];

        xr = (float)(x2-x1);
        yr = (float)(y2-y1);

        d = sqrt(pow(xr, 2) + pow(yr, 2));
        if(d!=0)
        {
            if (d > 1.5)
            {
                XF.push_back(x1);
                YF.push_back(y1);
                while(d > 1.5)
                {
                    ang = atan2(yr,xr) * 180 / pi;
                    if (ang<0){ang = ang + 360;}
                    d = angXY(ang, x1, y1, x2, y2, x_temp, y_temp);
                    XF.push_back(x_temp);
                    YF.push_back(y_temp);

                    xr = (float)(x2-x_temp);
                    yr = (float)(y2-y_temp);
                }
            }
            else
            {
                XF.push_back(X[kl]);
                YF.push_back(Y[kl]);
            }
        }
    }
    XF.push_back(X[kl]);
    YF.push_back(Y[kl]);
}

std::string ZeroPadNumber(int num, int zero_space)
{
	std::stringstream ss;

	// the number is converted to string with the help of stringstream
	ss << num;
	std::string ret;
	ss >> ret;

	// Append zero chars
	int str_length = ret.length();
	for (int i = 0; i < zero_space - str_length; i++)
		ret = "0" + ret;
    return ret;
}

void Simulate(std::string path_tracks, std::string path_backgrounds, std::string newSubFolder, int worms, int parall, int coll, int velocity, int day_init_exp, int total_image)
{   
    // Read info tracks
    std::vector<float> media_colors;
    std::vector<int> tam_velocity, r_xml;
    std:: string path_data_track = path_tracks + "tracks_info.txt";
    std::vector<std::string> xml_names = read_trackInfo(path_data_track,media_colors,tam_velocity,r_xml);

    // Read info backgrounds
    std::vector<int> Brc, Brx, Bry;
    std:: string path_data_backgrounds = path_backgrounds + "backgrounds_info.txt";
    std::vector<std::string> backgrounds_names = read_backgrounds_Info(path_data_backgrounds,Brc,Brx,Bry);

    srand (time(NULL));
    // Variable struct worm
    std::vector<Worm> data_worms;
    std::vector<std::vector<int>> XYn;
    std::vector<std::string> points;
    std::vector<int> total;
    std::vector<int> XYI;
    std::vector<int> random_vec;
    int speed, X, Y, X_offset, Y_offset;
    std::vector<int> XOFF, YOFF, XF, YF;
    std::vector<float> weights_scale;
    std::vector<float> colors_scale;
    std::vector<float> MA;
    
    // Variables to save pts file
    std::string XYP, XYP_Colors, XYP_Weights, selected_background;
    std::ofstream myfile, mybackground;
    std::string name_image, newSubFolderF, name_background, text_background;

    int random0, st, m, kl, c, ii, randon_background, R, CC1, CC2, check, pixelValue;
    std::vector<std::vector<int>> posI1( worms , std::vector<int> (2));

    //background selection
    selected_background="";
    randon_background = rand()%(backgrounds_names.size());
    selected_background = backgrounds_names[randon_background];
    R = Brc[randon_background];
    CC1 = Brx[randon_background];
    CC2 = Bry[randon_background];
                
    newSubFolderF = "";
    newSubFolderF = newSubFolder + "day = " + std::to_string(day_init_exp) + " cond = X pla = 1 living_worms = " + std::to_string(worms) + "/";
//     check = mkdir(newSubFolderF.c_str());
    check = mkdir(newSubFolderF.c_str(),S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH); //ubuntu<19.04

    // Create background file
    name_background="";
    text_background = "";
    name_background = newSubFolderF + "background";
    mybackground.open(name_background);
    text_background = selected_background + "," + std::to_string(R) + "," + std::to_string(CC1) + "," + std::to_string(CC2);
    mybackground << text_background << "\n";
    mybackground.close();
    
    std::string path_background = path_backgrounds + selected_background;
    cv::Mat img_background = cv::imread(path_background);

    data_worms.clear();
    std::vector<std::vector<int>> vec_worms;
    data_worms = get_data_worms(vec_worms, total_image, velocity, worms, path_tracks, xml_names, media_colors, tam_velocity, r_xml, R, CC1, CC2, parall, coll);

    for(ii=0;ii<data_worms.size();ii++)
    {
        random_vec.push_back(0);
    }

    std::string number_file;

    for(m=0;m<total_image;m++)
    {
        number_file = "";
        number_file = ZeroPadNumber(m+1, 2);

        name_image="";
        name_image = newSubFolderF + "RASP06_0000" + number_file + ".pts";
        myfile.open(name_image);

        for(ii=0;ii<data_worms.size();ii++)
        {
            weights_scale.clear();
            colors_scale.clear();
            points.clear();
            total.clear();
            XYn.clear();
            MA.clear();

            XOFF.clear();
            YOFF.clear();
            XF.clear();
            YF.clear();
            
            speed = data_worms[ii].Velocity;
            points = data_worms[ii].points;
            total = data_worms[ii].total;
            MA = data_worms[ii].MA;
            //XYn = get_data(points[m*speed],total[m*speed]);

            //Get first position skeleton
            if (m==0)
            {
                XYI.clear();
                XYI = data_worms[ii].XYI;
                posI1[ii][0] = XYI[0];
                posI1[ii][1] = XYI[1];
            }

            if((vec_worms[ii][0]==3) & (vec_worms[ii][1]==0))
            {
                if (m==0){random_vec[ii] = rand()%(12 - 8) + 8 +1;}
                if (m>=0 & m<=random_vec[ii]){XYn = get_data(points[random_vec[ii]*speed],total[random_vec[ii]*speed]);}
                else{XYn = get_data(points[m*speed],total[m*speed]);}
            }
            else {XYn = get_data(points[m*speed],total[m*speed]);}

            // Get rotate worm
            for (kl = 0; kl < XYn.size(); kl++)
            {
                X = XYn[kl][0] - posI1[ii][0];
                Y = XYn[kl][1] - posI1[ii][1];

                X_offset = round(((float)X*MA[0]) + ((float)Y*MA[1]) + data_worms[ii].Rx);
                Y_offset = round(((float)X*MA[2]) + ((float)Y*MA[3]) + data_worms[ii].Ry);

                XOFF.push_back(X_offset);
                YOFF.push_back(Y_offset);
            }

            check_XY(XOFF, YOFF, XF, YF);
            //Scale weights and colors
            scale_data(data_worms[ii].weights, data_worms[ii].colors, XF.size(), weights_scale, colors_scale);

            // Graph worm on image
            for (kl = 0; kl < XF.size(); kl++)
            {
                if(kl==0)
                {
                    XYP = "P";
                    XYP_Colors = "C";
                    XYP_Weights = "W";
                }

                pixelValue = (int)img_background.at<uchar>(XF[kl],YF[kl]);

                XYP = XYP + std::to_string(XF[kl]) + " " + std::to_string(YF[kl]);
                XYP_Colors = XYP_Colors + std::to_string((colors_scale[kl]+pixelValue)/2);
                XYP_Weights = XYP_Weights + std::to_string(weights_scale[kl]+0.5);

                if(kl!=XF.size()-1)
                {
                    XYP = XYP + ";";
                    XYP_Colors = XYP_Colors + ";";
                    XYP_Weights = XYP_Weights + ";";
                }
            }
            myfile << XYP << "\n";
            myfile << XYP_Colors << "\n";
            myfile << XYP_Weights << "\n\n";
        }
        myfile.close();
    }
    
    img_background.release();


    media_colors.clear();
    tam_velocity.clear();
    r_xml.clear();
    xml_names.clear();

    Brc.clear();
    Brx.clear();
    Bry.clear();
    backgrounds_names.clear();

    data_worms.clear();
    XYn.clear();
    points.clear();
    total.clear();
    XYI.clear();
    random_vec.clear();
    XOFF.clear();
    YOFF.clear();
    XF.clear();
    YF.clear();
    weights_scale.clear();
    colors_scale.clear();
    MA.clear();
}
