#ifdef _WIN64
#include "stdafx.h"
#include "windows.h"
// anything before a precompiled header is ignored, 
// so no endif here! add #endif to compile on __unix__ !
//#endif
#ifdef _WIN64
//#include <qhyccd.h>
#endif


/*
* From 
* https://stackoverflow.com/questions/15957671/opencv-gpudft-distorted-image-after-inverse-transform
*
*/

//#define _WIN64
//#define __unix__

#include <stdio.h>
#include <stdlib.h>

#ifdef __unix__
#include <unistd.h>

#endif

#include <string.h>

#include <time.h>
#include <sys/stat.h>
// this is for mkdir

#include <opencv2/opencv.hpp>
//#include <opencv2/gpu.hpp>        // GPU structures and methods
#include <opencv2/cudaarithm.hpp>	// which includes opencv2/core/cuda.hpp

// used the above include when imshow was being shown as not declared
// removing
// #include <opencv/cv.h>
// #include <opencv/highgui.h>

#include <iostream>
#include <sstream>

using namespace std;
///////////////////////////


using namespace cv;

void dftandinverseAsFunction(cuda::GpuMat lenaGPU, cuda::GpuMat lenaSpectrum, cuda::Stream stream, cuda::GpuMat lenaOut, double ticfreq, vector<cuda::GpuMat> splitter)
{
	double tic1, tic2;
	tic1 = (double)getTickCount();
	cuda::dft(lenaGPU, lenaSpectrum, lenaGPU.size(), 0, stream);
	//int c = lenaSpectrum.channels();
	//Size s = lenaSpectrum.size();
	cuda::dft(lenaSpectrum, lenaOut, lenaGPU.size(), DFT_INVERSE, stream);

	cuda::split(lenaOut, splitter, stream);
	stream.waitForCompletion();
	tic2 = (double)getTickCount();
	std::cout << "DFT and inverse, as a function " << (tic2 - tic1) / ticfreq << " sec." << std::endl;
}

int main()
{
	Mat lenaAfter;
	Mat lena = imread("bscanc001.png", IMREAD_GRAYSCALE);
	namedWindow("before", 1);    imshow("before", lena); 
	Mat lenaSpectrumCPU;
	Mat planescpu[2];
	Mat mag;

	lena.convertTo(lena, CV_32F, 1);

	std::vector<Mat> planes;
	planes.push_back(lena);
	planes.push_back(Mat::zeros(lena.size(), CV_32FC1));
	merge(planes, lena);

	cuda::GpuMat lenaGPU = cuda::GpuMat(512, 512, CV_32FC2);
	cuda::GpuMat lenaSpectrum = cuda::GpuMat(512, 512, CV_32FC2);
	cuda::GpuMat lenaOut = cuda::GpuMat(512, 512, CV_32FC2);
	cuda::Stream stream;

	vector<cuda::GpuMat>splitter;
	double tic1, tic2, tic3, tic4, ticfreq;
	ticfreq = (double)getTickFrequency();

	///////////////////////////////////
	// CPU version
	tic1 = (double)getTickCount();
	dft(lena, lenaSpectrumCPU);
	split(lenaSpectrumCPU, planescpu);
	magnitude(planescpu[0], planescpu[1], mag);
	dft(mag, lenaAfter, DFT_INVERSE);
	tic2 = (double)getTickCount();
	std::cout << "DFT and inverse, on CPU " << (tic2 - tic1) / ticfreq << " sec." << std::endl;
	imshow("afterCPU", lenaAfter); waitKey();

	///////////////////////////////////
	// GPU version
	tic1 = (double)getTickCount();

	lenaGPU.upload(lena);

	tic3 = (double)getTickCount();
	cuda::dft(lenaGPU, lenaSpectrum, lenaGPU.size(), 0, stream);
	//debug
	std::cout << "lenaGPU.size() = " << lenaGPU.size() << std::endl;
	int c = lenaSpectrum.channels();
	Size s = lenaSpectrum.size();
	//debug
	std::cout << "lenaSpectrum.channels(); = " << c << std::endl;
	std::cout << "lenaSpectrum.size(); = " << s << std::endl;

	cuda::dft(lenaSpectrum, lenaOut, lenaGPU.size(), DFT_INVERSE, stream);

	cuda::split(lenaOut, splitter, stream);
	stream.waitForCompletion();
	tic4 = (double)getTickCount();

	dftandinverseAsFunction(lenaGPU, lenaSpectrum, stream, lenaOut, ticfreq, splitter);
	dftandinverseAsFunction(lenaGPU, lenaSpectrum, stream, lenaOut, ticfreq, splitter);
	dftandinverseAsFunction(lenaGPU, lenaSpectrum, stream, lenaOut, ticfreq, splitter);
	dftandinverseAsFunction(lenaGPU, lenaSpectrum, stream, lenaOut, ticfreq, splitter);
	dftandinverseAsFunction(lenaGPU, lenaSpectrum, stream, lenaOut, ticfreq, splitter);
	dftandinverseAsFunction(lenaGPU, lenaSpectrum, stream, lenaOut, ticfreq, splitter);
	dftandinverseAsFunction(lenaGPU, lenaSpectrum, stream, lenaOut, ticfreq, splitter);
	dftandinverseAsFunction(lenaGPU, lenaSpectrum, stream, lenaOut, ticfreq, splitter);
	dftandinverseAsFunction(lenaGPU, lenaSpectrum, stream, lenaOut, ticfreq, splitter);
	dftandinverseAsFunction(lenaGPU, lenaSpectrum, stream, lenaOut, ticfreq, splitter);
	dftandinverseAsFunction(lenaGPU, lenaSpectrum, stream, lenaOut, ticfreq, splitter);

	// 2nd time
	tic3 = (double)getTickCount();
	cuda::dft(lenaGPU, lenaSpectrum, lenaGPU.size(), 0, stream);
	//c = lenaSpectrum.channels();
	//Size s = lenaSpectrum.size();
	cuda::dft(lenaSpectrum, lenaOut, lenaGPU.size(), DFT_INVERSE, stream);

	cuda::split(lenaOut, splitter, stream);
	stream.waitForCompletion();
	tic4 = (double)getTickCount();

	splitter[0].download(lenaAfter);
	//  lenaOut.download(lenaAfter);
	tic2 = (double)getTickCount();
	std::cout << "DFT and inverse, with upload/dl " << (tic2 - tic1) / ticfreq << " sec." << std::endl;
	std::cout << "DFT and inverse, without upload/dl " << (tic4 - tic3) / ticfreq << " sec." << std::endl;

	c = lenaAfter.channels();

	double n, x;
	minMaxIdx(lenaAfter, &n, &x);

	lenaAfter.convertTo(lenaAfter, CV_8U, 255.0 / x);
	namedWindow("after", 1);    imshow("after", lenaAfter); waitKey();
    return 0;
}

