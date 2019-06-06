#ifndef HEADALL_H_INCLUDED
#define HEADALL_H_INCLUDED

#include "opencv2/core/core.hpp"
#include "opencv2/ml/ml.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <cstdio>
#include <vector>
#include <iostream>
#include <chrono>
#include <ctime>
#include <stdlib.h>
#include <fstream>
#include <unistd.h>
#include <dirent.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <string.h>
#include <math.h>


/*MACRO*/
//#define Mpixel(image,x,y) (( uchar *)(((image).data)+(y)*((image).step)))[(x)]
#define pixelB( image, x, y) image.data[(y)*image.step[0]+(x)*image.step[1]]
#define pixelG( image, x, y) image.data[(y)*image.step[0]+(x)*image.step[1]+1]
#define pixelR( image, x, y) image.data[(y)*image.step[0]+(x)*image.step[1]+2]

#define  DEBUG_LOG()      cout << __FUNCTION__  << " line: "<< __LINE__ << endl;
#define PI      3.1415926

using namespace std;
using namespace cv;
using namespace chrono;
using namespace cv::ml;


//string filename_to_save =
//string data_filename =;
#define  DATA_FILENAME  "Trainfile.data"
#define  FILENAME_TO_SAVE   "Trainfile.xml"

char classarray[10]={'0','1','2','3','4','5','6','7','8','9'};
int classnumber=0;



#define  INPUT_PARAM_IMAGING   2
#define  INPUT_PARAM_CAMERA    1

float r;                //predict result
int r1;                 //similarity gesture with prediction
int upright=0;          //contour shape upright
Rect handposition;      //return contour rect area
float fps=0.0;          //fps of video
bool camera=0;          //parameter for the function with camera or without
int savefile=0;         //parameter to create data file of images
int FDFeatureNumber = 20; //change this for different FD, 19 in here (20-1)
Ptr<ANN_MLP> model;     //ANN classifier

#endif // HEADALL_H_INCLUDED
