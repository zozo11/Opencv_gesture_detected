/****************************************************************************************
* Zoe Wang
*
* g++ main.cpp -o main -std=c++11 `pkg-config --cflags --libs opencv`
*
* Train: ./main Trainfile.data 19
* Train file need over two parameters
*
* Test image: ./main ./img/Testimage.jpg
* Test image need to inpute two parameters input image file location and name
*
* Test Video: ./mian
****************************************************************************************/

#include "head/HeadAll.h"
#include  "training.cpp"

static vector< float> EllipticFourierDescriptors(vector<Point>& contour);
void Find_markers(Mat& image, Mat& output);
static int DetecImg(Mat img, string label1);
static void TransFD(Mat img, string label1);
static void ReadImage(const char *imgdir,const char *extension);
int CameraGesture(Mat img);
void CameraDetected();

//Fourier descriptors
static vector< float> EllipticFourierDescriptors(vector<Point>& contour)
{
    vector<float> ax,ay,bx,by;
    vector< float> CE;
    int m = contour.size() ;
    //number of CEs we are computing, +1 as the first number is always 2.0 and do not use it
    int n = FDFeatureNumber ;
    float t = (2*PI)/m;
    for(int k = 0;k < n; k++) {
        ax.push_back(0.0);
        ay.push_back(0.0);
        bx.push_back(0.0);
        by.push_back(0.0);
        for (int i=0;i<m;i++) {
            ax[k]=ax[k]+contour[i].x*cos((k+1)*t*(i));
            bx[k]=bx[k]+contour[i].x*sin((k+1)*t*(i));
            ay[k]=ay[k]+contour[i].y*cos((k+1)*t*(i));
            by[k]=by[k]+contour[i].y*sin((k+1)*t*(i));
        }
        ax[k]=(ax[k])/m;
        bx[k]=(bx[k])/m;
        ay[k]=(ay[k])/m;
        by[k]=(by[k])/m;
    }
    for(int k=0;k<n;k++) {
        CE.push_back(sqrt((ax[k]*ax[k]+ay[k]*ay[k])/(ax[0]*ax[0]+ay[0]*ay[0]))+sqrt((bx[k]*bx[k]+by[k]*by[k])/(bx[0]*bx[0]+by[0]*by[0])));
        //cout<<CE[k]<<endl;
    }
    return CE;
}

//Camera process
int CameraGesture(Mat img)
{
    Mat output;
    vector<vector<Point> > contours;
    vector<vector<Point> > contours0;
    vector<Vec4i> hierarchy;
    long int largestcontoursize=0;
    int largestcontour=0;

    Mat dstImg = Mat::zeros(img.rows, img.cols, CV_8UC3);
#if 0
    Mat mask(img.rows, img.cols, CV_8UC1);
    Mat temp1(img.rows,img.cols,CV_8UC1);
    Mat temp2(img.rows,img.cols,CV_8UC1);


    cvtColor(img,img_hsv,CV_BGR2HSV);
    //HSV image processing, to outline hands only
    inRange(img_hsv, Scalar(0,30,30), Scalar(40,170,256), temp1);
    inRange(img_hsv, Scalar(156,30,30), Scalar(180,170,256), temp2);
    bitwise_or(temp1, temp2, mask);
    //remove noise, edges processing
    Mat element = getStructuringElement(MORPH_RECT, Size(3,3));
    erode(mask, mask, element);
    morphologyEx(mask, mask, MORPH_OPEN, element);
    dilate(mask, mask, element);
    morphologyEx(mask, mask, MORPH_CLOSE, element)
#endif
    Find_markers(img, output);

    findContours(output, contours0, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE, Point());

    for(size_t k = 0;k < contours0.size(); k++) {
        if (largestcontoursize<contours0[k].size()) {
            largestcontoursize=contours0[k].size();
            largestcontour=k;
        }
    }

    contours.push_back(contours0[largestcontour]);


    approxPolyDP(Mat(contours0[largestcontour]), contours[0], 3, true);
    drawContours(dstImg, contours, 0, Scalar(0, 255, 255), 1, 8);  //draw contour

    if (savefile==0) imshow("contour",dstImg);


    vector<float> CE = EllipticFourierDescriptors(contours0[largestcontour]);

    Mat hand1 = Mat(Size(CE.size()-1,1),CV_32FC1,(void*)&CE[1]).clone();
    //get index value r
    r = model->predict(hand1);

    cout << "Prediction: " << r << " Gesture: " << classarray[(int)r] << endl;

    char str[25];

    snprintf(str, sizeof(str), "gesture:%d", classarray[(int)r]);

    putText(img,str,cvPoint(10, 60),CV_FONT_HERSHEY_DUPLEX,1.0f,CV_RGB(255,128,128));

    return r;
}

// Recognise gesture camera
void CameraDetected()
{
    Mat frame;//Image

    VideoCapture cap;
    cap.open(0);

    if (!cap.isOpened()) {
        cout << "Failed to open camera" << endl;
        return;
    }

    cout << "Opened camera" << endl;

    cap.set(CV_CAP_PROP_FRAME_WIDTH, 640);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, 480);

    cap >> frame;

    cout <<  "Frame size :  " <<  frame.rows << frame.cols << endl;

    namedWindow("Video Gestures",WINDOW_AUTOSIZE);

    while (true) {
        system_clock::time_point start = system_clock::now();

        cap >> frame;
        if( frame.empty() )
            break;

        int result = CameraGesture(frame);

        system_clock::time_point end = system_clock::now();
        double seconds = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        double fps = 1000000/seconds;

        cout << "frames: " << fps << " seconds: " << seconds << " gesture: "<< result << endl;

        char printit[100];
        sprintf(printit,"frames: %2.1f",fps);
        putText(frame, printit, cvPoint(10,30), FONT_HERSHEY_PLAIN, 2, cvScalar(128,128,128), 2, 8);

        imshow("Video Gestures",frame);

        int key = waitKey(1);
        if(key==113 || key==27) return;//either esc or 'q'
    }
}

//Test single image
void Find_markers(Mat& image, Mat& output){
 //convert to HSV, the order is the same as BGR
    const int upH=165; //137
    const int loH=40; //60
    const int upS=255;
    const int loS=70; //70
    const int upV=255;
    const int loV=98; //98

    int marker_upH=upH;
    int marker_loH=loH;
    int marker_upS=upS;
    int marker_loS=loS;
    int marker_upV=upV;
    int marker_loV=loV;

    Mat imageHSV;
    cvtColor(image,imageHSV, CV_RGB2HSV);
    output = Mat::zeros(image.rows, image.cols, CV_8UC1);

    //segment markers
	for (int x=0;x<imageHSV.cols;x++){
		for (int y=0;y<imageHSV.rows;y++){
			if( pixelR(imageHSV,x,y) < marker_loV || pixelR(imageHSV,x,y) > marker_upV ||
			pixelG(imageHSV,x,y) < marker_loS || pixelG(imageHSV,x,y) > marker_upS ||
			pixelB(imageHSV,x,y) < marker_loH || pixelB(imageHSV,x,y) > marker_upH   ){
				//pixelR(image,x,y)=0;
				//pixelG(image,x,y)=0;
				pixelB(output,x,y)=0;
			}
			else {
				//pixelR(image,x,y)=255;
				//pixelG(image,x,y)=255;
				pixelB(output,x,y)=255;
				//debugcount++;
			}
		}
	}
}


static int DetecImg(Mat img, string label1)
{
    Mat outImg;
    vector<float> CE;

    vector<vector<Point> > contours0;
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    long int largestcontoursize=0;
    int largestcontour=0;

    Mat dstImg = Mat::zeros(img.rows, img.cols, CV_8UC3);

    Find_markers(img, outImg);

    findContours( outImg, contours0, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE,Point());
        //save each contours in other vector for compare the contour point size
        for(size_t k=0;k<contours0.size();k++) {
            cout<<"Get contours size:"<<contours0[k].size()<<endl;
            if (largestcontoursize<contours0[k].size()) {
                largestcontoursize=contours0[k].size();
                largestcontour=k;
            }
        }
        contours.push_back(contours0[largestcontour]);
        //befor draw contours image get the edge for gesture
        approxPolyDP(Mat(contours0[largestcontour]), contours[0], 3, true);
        //draw contour by the point save in contours vector befor and the index will suing the contours number
        drawContours(dstImg, contours0, largestcontour, Scalar(0, 255, 255), 1, 8);

        CE=EllipticFourierDescriptors(contours0[largestcontour]);
        Mat hand1 = Mat(Size(CE.size()-1,1),CV_32FC1,(void*)&CE[1]).clone();

        r = model->predict(hand1);
        cout << " Image Prediction: " << r <<  endl;
        r=(int)r;
        //adjust index value according to the hand position
        char str[25];
        if (r1==0) sprintf(str,"Gesture:%c",classarray[(int)r]);
        else sprintf(str,"Gesture:%c/%c",classarray[(int)r],classarray[r1]);
        putText(img,str,cvPoint(10,30),FONT_HERSHEY_PLAIN,2,cvScalar(0,0,255),2,8);
        if (camera)
        {   sprintf(str,"%2.1f",fps);
            putText(img,str,cvPoint(10,80),FONT_HERSHEY_PLAIN,2,cvScalar(0,0,255),2,8);
        }

        imshow("Original",img);
        imshow("HSV Image",outImg);
        imshow("contour",dstImg);
        waitKey(0);
        return (int)r;
}

//Static image file
static void TransFD(Mat img, string label1)
{
    Mat img_gray;
    Mat kernel = (Mat_<int>(3, 3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);
    vector<vector<Point> > contours0;
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    long int largestcontoursize=0;
    int largestcontour=0;
    Mat dstImg = Mat::zeros(img.rows, img.cols, CV_8UC3);

#if 0
    filter2D(img, img, img.depth(), kernel);
    cvtColor(img,img_gray,CV_BGR2GRAY);
    threshold(img_gray,img_gray,230,255,CV_THRESH_TOZERO_INV);
    threshold(img_gray,img_gray,70,255,CV_THRESH_TOZERO);
    // Median blur to filter noise
    medianBlur(img_gray, img_gray, 7);
    Mat midImage = img_gray.clone();
    threshold(midImage,midImage,0,255,CV_THRESH_BINARY|CV_THRESH_OTSU);
#endif

    cvtColor(img,img_gray,CV_BGR2GRAY);
    threshold(img_gray, img_gray, 5, 255, CV_THRESH_BINARY);
//Debug the image
//imshow("threshold",img_gray);
//waitKey(0);
    //if (savefile==0) imshow("Binary image",midImage);

    //finder contours
    findContours( img_gray, contours0, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE,Point());
    //save each contours in other vector for compare the contour point size
    for(size_t k=0;k<contours0.size();k++) {
        cout<<"Get contours size:"<<contours0[k].size()<<endl;
        if (largestcontoursize<contours0[k].size()) {
            largestcontoursize=contours0[k].size();
            largestcontour=k;
        }
    }
    contours.push_back(contours0[largestcontour]);

#if 0
    //befor draw contours image get the edge for gesture
    approxPolyDP(Mat(contours0[largestcontour]), contours[0], 3, true);
    //draw contour by the point save in contours vector befor and the index will suing the contours number
    drawContours(dstImg, contours0, largestcontour, Scalar(0, 255, 255), 1, 8);
    if (savefile==0) imshow("contour",dstImg);
#endif
    vector<float> CE;
    //calc CE from an image
    CE=EllipticFourierDescriptors(contours0[largestcontour]);
    Mat hand1 = Mat(Size(CE.size()-1,1),CV_32FC1,(void*)&CE[1]).clone();

    cout << label1.length() << endl;

    if (label1!= "") {
        cout << "class: " << label1 <<" data: " << hand1 <<endl;
        ofstream datafile;// declaration of file pointer named datafile
        datafile.open("Trainfile.data", ios::app); // opens file named "xxxx" for output,add at the end
        datafile << label1;

        for (int i = 1; i < FDFeatureNumber;i++) {
            datafile << "," << CE[i];
        }
        datafile <<"\n";
        datafile.close();
    } else {
       /* //get index
        r = model->predict(hand1);
        cout << " Image Prediction: " << r <<  endl;
        r=(int)r;
        //adjust index value according to the hand position
        char str[25];
        snprintf(str, sizeof(str), "Gesture:%d", (int)r);
        putText(img,str,cvPoint(10,30),FONT_HERSHEY_PLAIN,2,cvScalar(0,0,255),2,8);
        if (camera)
        {   sprintf(str,"%2.1f",fps);
            putText(img,str,cvPoint(10,80),FONT_HERSHEY_PLAIN,2,cvScalar(0,0,255),2,8);
        }
        imshow("Original",img);
        waitKey(0);*/
        DetecImg(img,"");

    }
}

//Make data file from image
static void ReadImage(const char *imgdir,const char *extension)
{
    DIR *dir;
    struct dirent *ent;
    string image_filename;
    int i=0;
    if ((dir = opendir (imgdir)) != NULL) {
        while ((ent = readdir (dir)) != NULL) {
            if (strstr(ent->d_name,extension)!=NULL) {
                image_filename = ent->d_name;
                string label1;
                label1 = image_filename.substr(6,1).c_str();

                const char *cp = label1.c_str();

                if(cp[0] < 58 && cp[0] > 47 ) {
                    /*valid value*/
                    cout << "The valid image file:" << image_filename << endl;
                } else {
                    cout << "The invalide image file:" << image_filename << endl;
                    continue;
                }

                cout<<label1 <<":" <<  image_filename.substr(6,1).c_str();
                char imagefile[80];
                //concatenate file location
                sprintf(imagefile,"%s/%s",imgdir,image_filename.c_str());
                Mat img = imread(imagefile, 1);
                if( img.empty()) {
                    cout << "Couldn't load " <<imagefile << endl;
                } else {
                        TransFD(img,  label1);
                        i++;
                    }
                }
            }
        closedir(dir);
    }

    cout <<  "Total loaded image counter:" << i << endl;
}


int main(int argc, char *argv[])
{
    Mat img;

    string filename_to_load = "";

    /*do the traingin when the parameter > 2
     *  input any para  > 2  do this
     */
    if(argc > 2) {
        /* load the image and get the features*/
        ReadImage(argv[1],".png");
        cout<<"Train image"<<endl;
        /*do the training*/
        build_mlp_classifier(DATA_FILENAME, FILENAME_TO_SAVE,filename_to_load);
        return 0;
    }

    /*update the filename */
    filename_to_load =  FILENAME_TO_SAVE ;
    /*Loadding the value when the parameter =< 2*/
    if(!filename_to_load.empty()) {

        model = load_classifier<ANN_MLP>(filename_to_load);

        if(model.empty()){
            cout<<"need to train the mode for the first time"<<endl;
            exit(0);
        }
    }

    if (argc == INPUT_PARAM_IMAGING) {
        //Image
        cout << argv[1] << endl;

        img = imread(argv[1], 1);

        if( img.empty()) {
            cout << "Error: Couldn't load " << argv[1] << endl;
            exit(0);
        }

        DetecImg(img,"");
        waitKey(0);
        exit(0);
    } else {

        CameraDetected();
        return 0;
    }
        return 0;
}

