#!/bin/bash
#for testing scripts
rm main
g++ main.cpp -o main -std=c++11 `pkg-config --cflags --libs opencv`   > cmd.txt 2>&1
