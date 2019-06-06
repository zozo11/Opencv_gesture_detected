This is ReadMe file for Computer Vision

The Training file - Trainfile.xml Accuracy is 99.1%.

Using Gesturs Detected Program:

1. Compile file: g++ main.cpp -o main -std=c++11 `pkg-config --cflags --libs opencv`

2. Train Need input not over 3 parameters with traning images location and features number:  Training file (Trainfile.data and Trainfile.XML) :./main ./TrainingFile 19

3. Test single image file need input not over 2 parameters with test image file name and location:  ./main ./img/TestImage.jpg

4. Test Video only need run compile file: ./main
