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
* Compute offline Bscans, with CUDA.
* has option to save as Matlab readable file.
*
* Example Usage: ./Bscancompute.bin /path/to/Bscans 10 2
* The 2 is binvalue, optional, if the saved png files are different in size
* from the saved ocv files.
* The 10 is number of spectrogram images, in the filename style
* Trig001-000.png, Trig001-001.png, etc for the triggered captured spectrograms
* KTrig001-000.png, KTrig001-001.png, etc for the J0 null captured spectrograms
* and spectrum.ocv for the saved spectrum.
*
* n for next file, p for previous file
* s to save as
*
* ESC, x or X key quits
*
*
*
* Hari Nandakumar
* 28 Oct 2019  *
*
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
#include "tinyfiledialogs.h"

using namespace std;
///////////////////////////


using namespace cv;

inline void normalizerows(Mat& src, Mat& dst, double lowerlim, double upperlim)
{
	// https://stackoverflow.com/questions/10673715/how-to-normalize-rows-of-an-opencv-mat-without-a-loop
	// for syntax _OutputArray(B.ptr(i), B.cols))
	for (uint ii = 0; ii<src.rows; ii++)
	{
		normalize(src.row(ii), dst.row(ii), lowerlim, upperlim, NORM_MINMAX);
	}

}

inline void printAvgROI(Mat bscandb, uint ascanat, uint vertposROI, uint widthROI, Mat& ROIplot, uint& ROIploti, Mat& statusimg)
{
	Mat AvgROI;
	Scalar meanVal;
	uint heightROI = 3;
	char textbuffer[80];
	Mat lastrowofstatusimg = statusimg(Rect(0, 250, 600, 50)); // x,y,width,height

	if (ascanat + widthROI<bscandb.cols)
	{
		bscandb(Rect(ascanat, vertposROI, widthROI, heightROI)).copyTo(AvgROI);
		//imshow("ROI",AvgROI);
		AvgROI.reshape(0, 1);		// make it into a 1D array
		meanVal = mean(AvgROI);
		//printf("Mean of ROI at %d = %f dB\n", ascanat, meanVal(0));
		sprintf(textbuffer, "Mean of ROI at %d = %f dB", ascanat, meanVal(0));
		lastrowofstatusimg = Mat::zeros(cv::Size(600, 50), CV_64F);
		putText(statusimg, textbuffer, Point(0, 280), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 3, 1);
		imshow("Status", statusimg);
		// in ROIplot, we take the range to be 0 to 50 dB
		// this is mapped to 0 to 300 vertical pixels
		uint vertindex = uint(abs(6 * floor(meanVal(0))));

		if (vertindex < 300)
			vertindex = 300 - vertindex;	// since Mat is shown 0th row on top with imshow

		ROIplot.col(ROIploti) = Scalar::all(0);
		for (int smalloopi = -2; smalloopi < 4; smalloopi++)
		{
			if ((vertindex + smalloopi) > 0)
				if ((vertindex + smalloopi) < 300)
					ROIplot.at<double>(vertindex + smalloopi, ROIploti) = 1;
		}

		imshow("ROI intensity", ROIplot);
		if (ROIploti < 599)
			ROIploti++;
		else
			ROIploti = 0;
	}
	else
		sprintf(textbuffer, "ascanat+widthROI > width of image!");
	lastrowofstatusimg = Mat::zeros(cv::Size(600, 50), CV_64F);
	putText(statusimg, textbuffer, Point(0, 280), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 3, 1);
	imshow("Status", statusimg);
}

inline void printMinMaxAscan(Mat bscandb, uint ascanat, int numdisplaypoints, Mat& statusimg)
{
	Mat ascan, ascandisp;
	double minVal, maxVal;
	char textbuffer[80];
	Mat thirdrowofstatusimg = statusimg(Rect(0, 150, 600, 50));
	Mat fourthrowofstatusimg = statusimg(Rect(0, 200, 600, 50));
	bscandb.col(ascanat).copyTo(ascan);
	ascan.row(4).copyTo(ascan.row(1));	// masking out the DC in the display
	ascan.row(4).copyTo(ascan.row(0));
	ascan.row(4).copyTo(ascan.row(2));
	ascan.row(4).copyTo(ascan.row(3));
	ascandisp = ascan.rowRange(0, numdisplaypoints);
	//debug
	//normalize(ascan, ascandebug, 0, 1, NORM_MINMAX);
	//imshow("debug", ascandebug);
	minMaxLoc(ascandisp, &minVal, &maxVal);
	sprintf(textbuffer, "Max of Ascan%d = %f dB", ascanat, maxVal);
	thirdrowofstatusimg = Mat::zeros(cv::Size(600, 50), CV_64F);
	putText(statusimg, textbuffer, Point(0, 180), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 3, 1);
	sprintf(textbuffer, "Min of Ascan%d = %f dB", ascanat, minVal);
	fourthrowofstatusimg = Mat::zeros(cv::Size(600, 50), CV_64F);
	putText(statusimg, textbuffer, Point(0, 230), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 3, 1);
	imshow("Status", statusimg);

}

inline void makeonlypositive(Mat& src, Mat& dst)
{
	// from https://stackoverflow.com/questions/48313249/opencv-convert-all-negative-values-to-zero
	max(src, 0, dst);

}

inline cuda::GpuMat zeropadrowwise(cuda::GpuMat sm, int sn)
{
	// increase fft points sn times 
	// newnumcols = numcols*sn;
	// by fft, zero padding, and then inv fft

	// returns CV_32F for cuda

	// guided by https://stackoverflow.com/questions/10269456/inverse-fourier-transformation-in-opencv
	// inspired by Drexler & Fujimoto 2nd ed Section 5.1.10

	// needs fftshift implementation for the zero pad to work correctly if done on borders.
	// or else adding zeros directly to the higher frequencies. 

	// freqcomplex=fftshift(fft(signal));
	// zp2=4*ifft(ifftshift(zpfreqcomplex));

	// result of this way of zero padding in the fourier domain is to resample the same min / max range
	// at a higher sampling rate in the initial domain.
	// So this improves the k linear interpolation.

	//debug
	/*Mat temp;
	sm.download(temp);
	normalize(temp, temp, 0, 1, NORM_MINMAX);
	imshow("input to zp", temp);
	waitKey();*/

	cuda::GpuMat origimage;
	cuda::GpuMat fouriertransform, fouriertransformzp, zp0, zp1;
	cuda::GpuMat inversefouriertransform;
	cuda::GpuMat matzeros(sm.size(), CV_32FC1);
	matzeros.setTo(Scalar::all(0));
	cuda::Stream stream;

	int numrows = sm.rows;
	int numcols = sm.cols;
	int newnumcols = numcols * sn;

	//sm.convertTo(origimage, CV_32F);
	// we assume the GpuMat is already CV_32F, but we make it a complex input as shown below.
	std::vector<cuda::GpuMat> planes;
	std::vector<cuda::GpuMat> splitter;
	std::vector<cuda::GpuMat> zpplanes;

	planes.push_back(sm);
	planes.push_back(matzeros);
	cuda::merge(planes, origimage);

	//cuda::dft(origimage, fouriertransform, origimage.size(), DFT_SCALE | DFT_COMPLEX_OUTPUT | DFT_ROWS);
	// in cuda::dft, can't pass DFT_COMPLEX_OUTPUT
	// if DFT_COMPLEX_INPUT is not passed, output is compressed into half the size. 
	cuda::dft(origimage, fouriertransform, origimage.size(), DFT_SCALE | DFT_COMPLEX_INPUT | DFT_ROWS, stream);

	//debug
	//std::cout << "origimage.size, depth" << origimage.size() << "," << origimage.depth() << std::endl;
	//std::cout << "fouriertransform.size, depth" << fouriertransform.size() << "," << fouriertransform.depth() << std::endl;

	// implementing fftshift row-wise
	// like https://docs.opencv.org/2.4/doc/tutorials/core/discrete_fourier_transform/discrete_fourier_transform.html
	int cx = fouriertransform.cols / 2;

	// here we assume fouriertransform.cols is even

	cuda::GpuMat LHS(fouriertransform, Rect(0, 0, cx, fouriertransform.rows));   // Create a ROI per half
	cuda::GpuMat RHS(fouriertransform, Rect(cx, 0, cx, fouriertransform.rows)); //  Rect(topleftx, toplefty, w, h), 
																	   // OpenCV typically assumes that the top and left boundary of the rectangle are inclusive, while the right and bottom boundaries are not. 
																	   // https://docs.opencv.org/3.2.0/d2/d44/classcv_1_1Rect__.html

	cuda::GpuMat tmp;                           // swap LHS & RHS
	LHS.copyTo(tmp);
	RHS.copyTo(LHS);
	tmp.copyTo(RHS);
	//debug
	//std::cout << "fft-shift done" << std::endl;

	//debug
	//std::cout << "ft size, channels, depth" << fouriertransform.size() << "," << fouriertransform.channels() << "," << fouriertransform.depth() << std::endl;
	// cuda::copyMakeBorder does not support ft's CV32FC2 - only FC1 is supported. So, must split, do twice.
	cuda::split(fouriertransform, splitter, stream);

	//The following syntax does not work.
	//cuda::copyMakeBorder(splitter[0], zpplanes[0], 0, 0, floor((newnumcols - numcols) / 2), floor((newnumcols - numcols) / 2), BORDER_CONSTANT, Scalar(0.0, 0, 0, 0));
	//cuda::copyMakeBorder(splitter[1], zpplanes[1], 0, 0, floor((newnumcols - numcols) / 2), floor((newnumcols - numcols) / 2), BORDER_CONSTANT, Scalar(0.0, 0, 0, 0));
	// this does the zero pad - copyMakeBorder(src, dest, top, bottom, left, right, borderType, value)
	cuda::copyMakeBorder(splitter[0], zp0, 0, 0, floor((newnumcols - numcols) / 2), floor((newnumcols - numcols) / 2), BORDER_CONSTANT, Scalar(0.0, 0, 0, 0), stream);
	cuda::copyMakeBorder(splitter[1], zp1, 0, 0, floor((newnumcols - numcols) / 2), floor((newnumcols - numcols) / 2), BORDER_CONSTANT, Scalar(0.0, 0, 0, 0), stream);
	//debug
	//std::cout << "copy make border done" << std::endl;
	stream.waitForCompletion();
	zpplanes.push_back(zp0);
	zpplanes.push_back(zp1);
	
	cuda::merge(zpplanes, fouriertransformzp, stream);

	//debug
	//std::cout << "merge done" << std::endl;

	// Now we do the ifftshift before ifft
	cx = fouriertransformzp.cols / 2;
	cuda::GpuMat LHSzp(fouriertransformzp, Rect(0, 0, cx, fouriertransformzp.rows));   // Create a ROI per half
	cuda::GpuMat RHSzp(fouriertransformzp, Rect(cx, 0, cx, fouriertransformzp.rows)); //  Rect(topleftx, toplefty, w, h)

	LHSzp.copyTo(tmp);
	RHSzp.copyTo(LHSzp);
	tmp.copyTo(RHSzp);

	cuda::dft(fouriertransformzp, inversefouriertransform, fouriertransformzp.size(), DFT_COMPLEX_INPUT | DFT_INVERSE | DFT_REAL_OUTPUT | DFT_ROWS, stream);
	//inversefouriertransform.convertTo(inversefouriertransform, CV_64F);
	// in GpuMat, only CV_32F.
	stream.waitForCompletion();
	//debug
	//std::cout << "zp size, channels, depth" << inversefouriertransform.size() << "," << inversefouriertransform.channels() << "," << inversefouriertransform.depth() << std::endl;
	//debug
	//Mat temp;
	/*inversefouriertransform.download(temp);
	normalize(temp, temp, 0, 1, NORM_MINMAX);
	imshow("op to zp", temp);
	waitKey();*/

	return inversefouriertransform;
}

inline Mat smoothmovavg(Mat sm, int sn)
{
	// smooths each row of Mat m using 2n+1 point weighted moving average
	// x(p) = ( x(p-n) + x(p-n+1) + .. + 2*x(p) + x(p+1) + ... + x(p+n) ) / 2*(n+1)
	// The window size is truncated at the edges.

	// can see https://docs.opencv.org/2.4/doc/tutorials/core/how_to_scan_images/how_to_scan_images.html#howtoscanimagesopencv
	// for efficient ways 

	// accept only double type matrices
	// sm needs to be CV_64FC1
	CV_Assert(sm.depth() == CV_64F);

	Mat sresult;
	sm.copyTo(sresult);		// initializing size of result

	int smaxcols = sm.cols;
	int smaxrows = sm.rows;

	double ssum;
	int sindexi;
	double* srcptr;
	double* destptr;

	for (int si = 0; si < smaxrows; si++)
	{
		srcptr = sm.ptr<double>(si);
		destptr = sresult.ptr<double>(si);

		for (int sj = 0; sj < smaxcols; sj++)
		{
			ssum = 0;

			for (int sk = -sn; sk < (sn + 1); sk++)
			{
				// address as m.at<double>(y, x); ie (row,column)
				sindexi = sj + sk;
				if ((sindexi > -1) && (sindexi < smaxcols))	// truncate window 
					ssum = ssum + srcptr[sindexi];		//equivalent to ssum = ssum + sm.at<double>(si,sindexi);
				else
					ssum = ssum + srcptr[sj];				// when window is truncated,
															// weight of original point increases

			}

			// we want to add m.at<double>(i,j) once again, since its weight is 2
			ssum = ssum + srcptr[sj];
			destptr[sj] = ssum / 2 / (sn + 1);		//equivalent to sresult.at<double>(si,sj) = ssum / (2 * (sn+1) );

		}

	}

	return sresult;



}

// from http://stackoverflow.com/a/32357875/5294258
void matwrite(const string& filename, const Mat& mat)
{
	ofstream fs(filename, fstream::binary);

	// Header
	int type = mat.type();
	int channels = mat.channels();
	fs.write((char*)&mat.rows, sizeof(int));    // rows
	fs.write((char*)&mat.cols, sizeof(int));    // cols
	fs.write((char*)&type, sizeof(int));        // type
	fs.write((char*)&channels, sizeof(int));    // channels

												// Data
	if (mat.isContinuous())
	{
		fs.write(mat.ptr<char>(0), (mat.dataend - mat.datastart));
	}
	else
	{
		int rowsz = CV_ELEM_SIZE(type) * mat.cols;
		for (int r = 0; r < mat.rows; ++r)
		{
			fs.write(mat.ptr<char>(r), rowsz);
		}
	}
}

Mat matread(const string& filename)
{
	ifstream fs(filename, fstream::binary);

	// Header
	int rows, cols, type, channels;
	fs.read((char*)&rows, sizeof(int));         // rows
	fs.read((char*)&cols, sizeof(int));         // cols
	fs.read((char*)&type, sizeof(int));         // type
	fs.read((char*)&channels, sizeof(int));     // channels

												// Data
	Mat mat(rows, cols, type);
	fs.read((char*)mat.data, CV_ELEM_SIZE(type) * rows * cols);

	return mat;
}

inline void savematasbin(char* p, char* d, char* f, Mat m)
{
	// saves a Mat m by writing to a binary file  f appending .ocv, both windows and unix versions
	// p=pathname, d=dirname, f=filename

#ifdef __unix__
	strcpy(p, d);
	strcat(p, "/");
	strcat(p, f);
	strcat(p, ".ocv");
	matwrite(p, m);
#else

	strcpy(p, d);
	strcat(p, "\\");		// imwrite needs path with \\ separators, not /, on windows
	strcat(p, f);
	strcat(p, ".ocv");
	matwrite(p, m);
#endif	

}


inline void savematasimage(char* p, char* d, char* f, Mat m)
{
	// saves a Mat m using imwrite as filename f appending .png, both windows and unix versions
	// p=pathname, d=dirname, f=filename

#ifdef __unix__
	strcpy(p, d);
	strcat(p, "/");
	strcat(p, f);
	strcat(p, ".png");
	imwrite(p, m);
#else

	strcpy(p, d);
	strcat(p, "\\");		// imwrite needs path with \\ separators, not /, on windows
	strcat(p, f);
	strcat(p, ".png");
	imwrite(p, m);
#endif	

}


// the next function saves a Mat m as variablename f by dumping to outfile o, both windows and unix versions

#ifdef __unix__
inline void savematasdata(std::ofstream& o, char* f, Mat m)
{
	// saves a Mat m as variable named f in Matlab m file format
	o << f << "=";
	o << m;
	o << ";" << std::endl;
}

#else
inline void savematasdata(cv::FileStorage& o, char* f, Mat m)
{
	// saves Mat m by serializing to xml as variable named f
	o << f << m;
}
#endif

bool fexists(const char *filename)
{
	std::ifstream ifile(filename);
	return (bool)ifile;
}

Mat linearbscan(Mat data_y, Mat data_yb, Mat data_ylin, Mat barthannwin, int increasefftpointsmultiplier, int numdisplaypoints, Mat diffk, Mat slopes, Mat fractionalk, Mat nearestkindex)
{
	cuda::GpuMat gdata_y;
	cuda::GpuMat gdata_yapod;
	cuda::GpuMat gdata_yoffset;
	cuda::GpuMat gdata_yb;
	cuda::GpuMat gdata_ylin;
	cuda::GpuMat gbarthannwin;
	cuda::GpuMat gdiffk;
	cuda::GpuMat gslopes;

	double tic3, tic4, ticfreq;
	ticfreq = (double)getTickFrequency();

	//cuda::GpuMat gfractionalk;
	//cuda::GpuMat gnearestkindex;
	cuda::GpuMat gbscantemp;

	gdata_y.upload(data_y);
	gdata_yb.upload(data_yb);
	//gdata_ylin.upload(data_ylin);
	gbarthannwin.upload(barthannwin);
	gdiffk.upload(diffk);
	gslopes.upload(slopes);
	//gfractionalk.upload(fractionalk);
	//gnearestkindex.upload(nearestkindex);

	// apodize 
	// data_y = ( (data_y - data_yb) ./ data_yb ).*gausswin
	cuda::Stream stream;
	cuda::divide(gdata_y, gdata_yb, gdata_yapod, 1, -1, stream);
	//debug
	//cout << "(data_y ) / data_yb done" << endl;
	//savematasdata
	//~ std::ofstream datay("datay.m");
	//~ datay << "datay" << "=";
	//~ datay << data_y;
	//~ datay << ";" << std::endl;


	for (int p = 0; p<(gdata_yapod.rows); p++)
	{
		//DC removal
		Scalar meanval = cuda::sum(gdata_yapod.row(p));
		meanval = meanval / gdata_yapod.cols;
		cuda::subtract(gdata_yapod.row(p), meanval(0), gdata_y.row(p), noArray(), -1, stream);	
		// Only the first value of the scalar is useful for us

														//windowing
		cuda::multiply(gdata_y.row(p), gbarthannwin, gdata_yapod.row(p), 1, -1, stream);
		
	}
	// debug
	//std::cout << "subtract, multiply done." << std::endl;
	//~ std::ofstream dataywin("dataywin.m");
	//~ dataywin << "dataywin" << "=";
	//~ dataywin << data_y;
	//~ dataywin << ";" << std::endl;

	//increasing number of points by zero padding
	tic3 = (double)getTickCount();
	if (increasefftpointsmultiplier > 1)
		gdata_y = zeropadrowwise(gdata_yapod, increasefftpointsmultiplier);
	tic4 = (double)getTickCount();
	std::cout << "zeropadrowwise " << (tic4 - tic3) / ticfreq << " sec." << std::endl;
	
	// debug
	//std::cout << "zeropad done." << std::endl;
	//debug
	//std::cout << "gdata_y size, channels, depth" << gdata_y.size() << "," << gdata_y.channels() << "," << gdata_y.depth() << std::endl;

	// interpolate to linear k space 
	// for doing data_y.at<double>(p, q) - data_y.at<double>(p, q - 1);
	// create an offset matrix
	//cuda::rshift(gdata_y, Scalar(1, 0, 0, 0), gdata_yoffset, stream); does not work for 32F,
	// moreover, it does bitwise shifting
	//gdata_y.copyTo(gdata_yoffset);	// initialize
	//for (int q = 1; q < (gdata_y.cols); q++)
	//{
	//	gdata_yoffset.col(q) = gdata_y.col(q - 1);
	//}

	// debug
	//std::cout << "elementwise rshift done." << std::endl;
	//debug
	//std::cout << "gdata_yoffset size, channels, depth" << gdata_yoffset.size() << "," << gdata_yoffset.channels() << "," << gdata_yoffset.depth() << std::endl;

	//cuda::subtract(gdata_y, gdata_yoffset, gslopes, noArray(), -1, stream);
	// debug
	//std::cout << "subtr done." << std::endl;
	//gslopes.col(0) = gslopes.col(1);
	// debug
	//std::cout << "ini col 1 done." << std::endl;
	

	//for (int p = 0; p<(gdata_y.rows); p++)
	//{
	//	for (int q = 1; q<(data_y.cols); q++)
	//	{
	//		//find the slope of the data_y at each of the non-linear ks
	//		slopes.at<double>(p, q) = data_y.at<double>(p, q) - data_y.at<double>(p, q - 1);
	//		// in the .at notation, it is <double>(y,x)
	//		//printf("slopes(%d,%d)=%f \n",p,q,slopes.at<double>(p,q));
	//	}
	//	// initialize the first slope separately
	//	slopes.at<double>(p, 0) = slopes.at<double>(p, 1);
	//stream.waitForCompletion();
	//gslopes.download(slopes);
	gdata_y.download(data_y);
	//// debug
	////std::cout << "download done." << std::endl;
	//Mat temp;
	////inversefouriertransform.download(temp);
	//normalize(slopes, temp, 0, 1, NORM_MINMAX);
	//imshow("slopes", temp);
	//waitKey();
	//slopes has a bug, so using CPU version.
	// except that everything is 32F now and not 64F.

	
	for (int p = 0; p<(data_y.rows); p++)
	{
		for (int q = 1; q<(data_y.cols); q++)
		{
			//find the slope of the data_y at each of the non-linear ks
			slopes.at<float>(p, q) = data_y.at<float>(p, q) - data_y.at<float>(p, q - 1);
			// in the .at notation, it is <double>(y,x)
			//printf("slopes(%d,%d)=%f \n",p,q,slopes.at<float>(p,q));
		}
		// initialize the first slope separately
		slopes.at<float>(p, 0) = slopes.at<float>(p, 1);


		for (int q = 1; q<(data_ylin.cols - 1); q++)
		{
			//find the value of the data_ylin at each of the klinear points
			// data_ylin = data_y(nearestkindex) + fractionalk(nearestkindex)*slopes(nearestkindex)
			//std::cout << "q=" << q << " nearestk=" << nearestkindex.at<int>(0,q) << std::endl;
			data_ylin.at<float>(p, q) = data_y.at<float>(p, nearestkindex.at<int>(0, q))
				+ fractionalk.at<float>(nearestkindex.at<int>(0, q))
				* slopes.at<float>(p, nearestkindex.at<int>(0, q));
			//printf("data_ylin(%d,%d)=%f \n",p,q,data_ylin.at<float>(p,q));
		}
		//data_ylin.at<double>(p, 0) = 0;
		//data_ylin.at<double>(p, numfftpoints) = 0;

	}

	//data_ylin.convertTo(data_ylin, CV_32F); - it is already 32F
	gdata_ylin.upload(data_ylin);
	// debug
	//std::cout << "upload done." << std::endl;
	//debug
	//~ std::ofstream dataylin("dataylin.m");
	//~ dataylin << "dataylin" << "=";
	//~ dataylin << data_ylin;
	//~ dataylin << ";" << std::endl;

	// InvFFT

	//cuda::GpuMat gplanes[] = { cuda::GpuMat_<float>(gdata_ylin), cuda::GpuMat::zeros(gdata_ylin.size(), CV_32F) };
	// this syntax  needs to be changed, to the push_back syntax
	cuda::GpuMat gcomplexI, gmagI, gcomplexIout;
	vector<cuda::GpuMat> gplanes, gplanesIn;
	cuda::GpuMat matzeros(gdata_ylin.size(), CV_32FC1);
	gplanesIn.push_back(gdata_ylin);
	gplanesIn.push_back(matzeros);

	cuda::merge(gplanesIn, gcomplexI);         // Add to the expanded another plane with zeros

	cuda::dft(gcomplexI, gcomplexIout, gdata_ylin.size(), DFT_COMPLEX_INPUT | DFT_ROWS | DFT_INVERSE, stream);
	// debug
	//std::cout << "inv dft done." << std::endl;
	cuda::split(gcomplexIout, gplanes, stream);                   // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
	stream.waitForCompletion();
	cuda::magnitude(gplanes[0], gplanes[1], gmagI);

	gbscantemp = gmagI.colRange(0, numdisplaypoints);
	Mat bscantemp;
	
	gbscantemp.download(bscantemp);
	//debug
	//std::cout << "download done" << std::endl;
	//std::cout << "gbscantemp size, channels, depth" << gbscantemp.size() << "," << gbscantemp.channels() << "," << gbscantemp.depth() << std::endl;

	bscantemp.convertTo(bscantemp, CV_64F);
	//debug
	//~ std::ofstream bscant("bscant.m");
	//~ bscant << "bscant" << "=";
	//~ bscant << bscantemp;
	//~ bscant << ";" << std::endl;
	return bscantemp;
}

int main(int argc, char *argv[])
{
	
	int num = 0;
	//qhyccd_handle *camhandle = NULL;
	int ret;
	char id[32];
	//char camtype[16];
	int found = 0;
	unsigned int w, h, bpp = 16, channels, cambitdepth = 16;
	unsigned int offsetx = 0, offsety = 0;
	unsigned int indexi, manualindexi, averages = 1, opw, oph;
	uint  indextemp;
	//uint indextempl;
	uint ascanat = 20;
	uint averagestoggle = 1;


	int camtime = 1, camgain = 1, camspeed = 1, cambinx = 2, cambiny = 2, usbtraffic = 10;
	int camgamma = 1, binvalue = 2, normfactor = 1, normfactorforsave = 25;
	int numfftpoints = 2880;
	int numdisplaypoints = 360;
	bool saveframes = 0;
	bool manualaveraging = 0, saveinterferograms = 0;
	unsigned int manualaverages = 1;
	int movavgn = 0;
	bool clampupper = 0;
	bool ROIreport = 0;

	bool doneflag = 0, skeypressed = 0, nkeypressed = 0, pkeypressed = 0;

	Mat jmask, jmaskt;
	double lambdamin, lambdamax;
	lambdamin = 840.5e-9;
	lambdamax = 859.5e-9;
	int mediann = 0;
	uint increasefftpointsmultiplier = 4;
	double bscanthreshold = -30.0;
	bool rowwisenormalize = 0;
	bool donotnormalize = 1;

	w = 1440;
	h = 960;

	int  fps, key;
	int t_start, t_end;
	double tic1, tic2, ticfreq;
	ticfreq = (double)getTickFrequency();
	tic1 = (double)getTickCount();

	//t_start = time(NULL);

	char dirname[80];
	char filename[20];
	char filenamec[20];
	char pathname[140];
	char lambdamaxstr[40];
	char lambdaminstr[40];

	if (argc > 3)	// then we have input the offline binning parameter as well
		binvalue = atoi(argv[3]);

	struct tm *timenow;

	time_t now = time(NULL);



	namedWindow("offline Bscan", 0); // 0 = WINDOW_NORMAL
	moveWindow("offline Bscan", 900, 500);



	/*Mat bscanmanualsave0[100];
	Mat bscanmanualsave1[100];

	Mat interferogramsave0[100];
	Mat interferogramsave1[100];
	Mat interferogrambsave0[100];
	Mat interferogrambsave1[100];*/

	/////////////////////////////////////
	// init camera, variables, etc

	cambitdepth = bpp;
	opw = w / binvalue;
	oph = h / binvalue;
	float lambda0 = (lambdamin + lambdamax) / 2;
	float lambdabw = lambdamax - lambdamin;

	unsigned int vertposROI = 10, widthROI = 10;

	Mat data_y(oph, opw, CV_64F);		// the Mat constructor Mat(rows,columns,type)
	Mat data_ylin(oph, numfftpoints, CV_32F);	 // changed for cuda
	Mat data_yb(oph, opw, CV_64F);
	Mat data_yp(oph, opw, CV_64F);
	Mat padded, paddedn;
	Mat barthannwin(1, opw, CV_64F);		// the Mat constructor Mat(rows,columns,type);
	Mat baccum, manualaccum;
	uint baccumcount, manualaccumcount;

	// initialize data_yb with zeros
	data_yb = Mat::zeros(Size(opw, oph), CV_64F);		//Size(cols,rows)		
	data_yp = Mat::zeros(Size(opw, oph), CV_64F);
	baccum = Mat::zeros(Size(opw, oph), CV_64F);
	baccumcount = 0;

	manualaccumcount = 0;

	Mat bscansave[100];		// allocate buffer to save linear bscans, max 100
	Mat jscansave[100];		// and jscans

							//Mat jscansave;		// to save j frames


	Mat bscansublog, positivediff;

	bool zeroisactive = 1;

	//int nr, nc;

	Mat m, opm, opmvector, bscan, bscanlog, bscandb, bscandisp, bscandispmanual, bscantemp, bscantemp2, bscantemp3, bscantransposed, chan[3];
	Mat tempmat;
	Mat rgbchannels[3];
	Mat bscandispj;
	Mat mraw;
	Mat ROIplot = Mat::zeros(cv::Size(600, 300), CV_64F);
	uint ROIploti = 0;
	Mat statusimg = Mat::zeros(cv::Size(600, 300), CV_64F);
	Mat firstrowofstatusimg = statusimg(Rect(0, 0, 600, 50)); // x,y,width,height
	Mat secrowofstatusimg = statusimg(Rect(0, 50, 600, 50));
	Mat secrowofstatusimgRHS = statusimg(Rect(300, 50, 300, 50));
	char textbuffer[80];

	//Mat bscanl, bscantempl, bscantransposedl;
	Mat magI, cmagI, cmagImanual;
	//Mat magIl, cmagIl;
	//double minbscan, maxbscan;
	//double minbscanl, maxbscanl;
	Scalar meanval;
	Mat lambdas, k, klinear;
	Mat diffk, slopes, fractionalk, nearestkindex;

	double kmin, kmax;
	double pi = 3.141592653589793;

	double minVal, maxVal;
	Mat ascan;
	//minMaxLoc( m, &minVal, &maxVal, &minLoc, &maxLoc );

	double deltalambda = (lambdamax - lambdamin) / data_y.cols;


	klinear = Mat::zeros(cv::Size(1, numfftpoints), CV_64F);
	fractionalk = Mat::zeros(cv::Size(1, numfftpoints), CV_64F);
	nearestkindex = Mat::zeros(cv::Size(1, numfftpoints), CV_32S);

	if (increasefftpointsmultiplier > 1)
	{
		lambdas = Mat::zeros(cv::Size(1, increasefftpointsmultiplier*data_y.cols), CV_64F);		//Size(cols,rows)
		diffk = Mat::zeros(cv::Size(1, increasefftpointsmultiplier*data_y.cols), CV_64F);
		slopes = Mat::zeros(cv::Size(data_y.rows, increasefftpointsmultiplier*data_y.cols), CV_64F);
	}
	else
	{
		lambdas = Mat::zeros(cv::Size(1, data_y.cols), CV_64F);		//Size(cols,rows)
		diffk = Mat::zeros(cv::Size(1, data_y.cols), CV_64F);
		slopes = Mat::zeros(cv::Size(data_y.rows, data_y.cols), CV_64F);

	}

	//resizeWindow("Bscan", oph, numdisplaypoints);		// (width,height)

	for (indextemp = 0; indextemp<(increasefftpointsmultiplier*data_y.cols); indextemp++)
	{
		// lambdas = linspace(830e-9, 870e-9 - deltalambda, data_y.cols)
		lambdas.at<double>(0, indextemp) = lambdamin + indextemp * deltalambda / increasefftpointsmultiplier;

	}
	k = 2 * pi / lambdas;
	kmin = 2 * pi / (lambdamax - deltalambda);
	kmax = 2 * pi / lambdamin;
	double deltak = (kmax - kmin) / numfftpoints;

	for (indextemp = 0; indextemp<(numfftpoints); indextemp++)
	{
		// klinear = linspace(kmin, kmax, numfftpoints)
		klinear.at<double>(0, indextemp) = kmin + (indextemp + 1)*deltak;
	}



	//for (indextemp=0; indextemp<(data_y.cols); indextemp++) 
	//{
	//printf("k=%f, klin=%f\n", k.at<double>(0,indextemp), klinear.at<double>(0,indextemp));
	//}


	for (indextemp = 1; indextemp<(increasefftpointsmultiplier*data_y.cols); indextemp++)
	{
		//find the diff of the non-linear ks
		// since this is a decreasing series, RHS is (i-1) - (i)
		diffk.at<double>(0, indextemp) = k.at<double>(0, indextemp - 1) - k.at<double>(0, indextemp);
		//printf("i=%d, diffk=%f \n", indextemp, diffk.at<double>(0,indextemp));
	}
	// and initializing the first point separately
	diffk.at<double>(0, 0) = diffk.at<double>(0, 1);

	for (int f = 0; f < numfftpoints; f++)
	{
		// find the index of the nearest k value, less than the linear k
		for (indextemp = 0; indextemp < increasefftpointsmultiplier*data_y.cols; indextemp++)
		{
			//printf("Before if k=%f,klin=%f \n",k.at<double>(0,indextemp),klinear.at<double>(0,f));
			if (k.at<double>(0, indextemp) < klinear.at<double>(0, f))
			{
				nearestkindex.at<int>(0, f) = indextemp;
				//printf("After if k=%f,klin=%f,nearestkindex=%d\n",k.at<double>(0,indextemp),klinear.at<double>(0,f),nearestkindex.at<int>(0,f));
				break;

			}	// end if


		}		//end indextemp loop

	}		// end f loop

	for (int f = 0; f < numfftpoints; f++)
	{
		// now find the fractional amount by which the linearized k value is greater than the next lowest k
		fractionalk.at<double>(0, f) = (klinear.at<double>(0, f) - k.at<double>(0, nearestkindex.at<int>(0, f))) / diffk.at<double>(0, nearestkindex.at<int>(0, f));
		//printf("f=%d, klinear=%f, diffk=%f, k=%f, nearesti=%d\n",f, klinear.at<double>(0,f), diffk.at<double>(0,nearestkindex.at<int>(0,f)), k.at<double>(0,nearestkindex.at<int>(0,f)),nearestkindex.at<int>(0,f) );
		//printf("f=%d, fractionalk=%f\n",f, fractionalk.at<double>(0,f));
	}

	averages = atoi(argv[2]);
	strcpy(pathname, argv[1]);
	strcat(pathname, "/spectrum.ocv");
	if (fexists(pathname))
		data_yb = matread(pathname);
	else
	{
		cout << "No saved spectrum found, offline tool exiting." << endl;
		return 1;
	}

	// debug
	//cout << "spectrum read " << data_yb.cols << "x" << data_yb.rows << endl;

	indextemp = 0;
	indexi = 1;
	averagestoggle = averages;

	bscantransposed = Mat::zeros(Size(numdisplaypoints, oph), CV_64F);
	// for testing, as below
	//bscantransposed = Mat::zeros(Size(opw, oph), CV_64F);
	bscantransposed.copyTo(bscantemp3);
	bscantransposed.copyTo(bscantemp2);

	for (uint p = 0; p<(opw); p++)
	{
		// create modified Bartlett-Hann window
		// https://in.mathworks.com/help/signal/ref/barthannwin.html
		float nn = p;
		float NN = opw - 1;
		barthannwin.at<double>(0, p) = 0.62 - 0.48*std::abs(nn / NN - 0.5) + 0.38*std::cos(2 * pi*(nn / NN - 0.5));

	}

	//t_end = time(NULL);

	//All these need to be CV_32F

	//gdata_y.upload(data_y);
	//gdata_yb.upload(data_yb);
	////gdata_ylin.upload(data_ylin);
	//gbarthannwin.upload(barthannwin);
	//gdiffk.upload(diffk);
	//gslopes.upload(slopes);
	////gfractionalk.upload(fractionalk);
	////gnearestkindex.upload(nearestkindex);

	//data_y.convertTo(data_y, CV_32F); inside loop
	//data_yb.convertTo(data_yb, CV_32F);
	barthannwin.convertTo(barthannwin, CV_32F);
	diffk.convertTo(diffk, CV_32F);
	slopes.convertTo(slopes, CV_32F);
	fractionalk.convertTo(fractionalk, CV_32F);

	tic2 = (double)getTickCount();
	std::cout << "Before while loop " << (tic2-tic1)/ticfreq <<" sec."<<std::endl;
	while (1)
	{

		tic1 = (double)getTickCount();
		while (indextemp < averages)
		{
			strcpy(pathname, argv[1]);
			sprintf(filename, "/Trig%03d-%03d.png", indexi, indextemp);
			strcat(pathname, filename);
			if (fexists(pathname))
			{
				opm = imread(pathname);
				split(opm, rgbchannels);
				//debug
				//cout << filename << " read" << endl;
			}
			else
			{
				cout << "A Trig file is missing, offline tool exiting." << endl;
				return 1;
			}


			/////////////////////////////////////////////////////////
			// do all Bscan processing here

			rgbchannels[0].convertTo(data_y, CV_32F);	// initialize data_y
			if (argc > 3)	// then we have input the offline binning parameter as well
				resize(data_y, data_y, Size(), 1.0 / binvalue, 1.0 / binvalue, INTER_AREA);	// binning (averaging)

			if (data_yb.channels() > 1)
			{
				split(data_yb, rgbchannels);
				rgbchannels[0].convertTo(data_yb, CV_32F);

			} 
			else
				data_yb.convertTo(data_yb, CV_32F);
			//debug
			//cout << "converted to data_y" << endl;
			//cout << "data_y size, channels, depth = " << data_y.size() << ", " << data_y.channels() << ", " << data_y.depth();
			//cout << "data_yb size, channels, depth = " << data_yb.size() << ", " << data_yb.channels() << ", " << data_yb.depth();

			//debug
			//return 0;

			bscansave[indextemp] = linearbscan(data_y, data_yb, data_ylin, barthannwin, increasefftpointsmultiplier, numdisplaypoints, diffk, slopes, fractionalk, nearestkindex);
			// for testing wo fft, as below
			//bscansave[indextemp] = data_y;
			//debug
			//cout << "bscanscave calculated" << endl;

			strcpy(pathname, argv[1]);
			sprintf(filename, "/KTrig%03d-%03d.png", indexi, indextemp);
			strcat(pathname, filename);
			if (fexists(pathname))
			{
				opm = imread(pathname);
				split(opm, rgbchannels);
			}
			else
			{
				cout << "A KTrig file is missing, offline tool exiting." << endl;
				return 1;
			}

			rgbchannels[0].convertTo(data_y, CV_32F);	// initialize data_y
			if (argc > 3)	// then we have input the offline binning parameter as well
				resize(data_y, data_y, Size(), 1.0 / binvalue, 1.0 / binvalue, INTER_AREA);	// binning (averaging)
																							/////////////////////////////////////////////////////////
																							// do all Bscan processing here
			jscansave[indextemp] = linearbscan(data_y, data_yb, data_ylin, barthannwin, increasefftpointsmultiplier, numdisplaypoints, diffk, slopes, fractionalk, nearestkindex);
			// for testing wo fft, as below
			//jscansave[indextemp] = data_y;
			// debug
			//cout << "jscansave is calculated" << endl;

			bscantemp3 = bscansave[indextemp] - jscansave[indextemp];
			// for testing wo fft, as below
			//bscantemp3 = bscansave[indextemp];
			
			// debug
			//cout << "bscantemp3 is calculated" << endl;

			makeonlypositive(bscantemp3, bscantemp2);
			bscantransposed = bscantransposed + bscantemp2;
			// debug
			//cout << "bscantransposed is calculated" << endl;

			indextemp++;

			//accumulate(bscantemp, bscantransposed);

		} // end while (indextemp < averages)

		tic2 = (double)getTickCount();
		std::cout << "While loop " << (tic2 - tic1) / ticfreq << " sec." << std::endl;


		  // this is the final display routine
		transpose(bscantransposed, bscan);
		//bscan = bscan / averagestoggle;
		bscan += Scalar::all(0.00001);   	// to prevent log of 0  
											// 20.0 * log(0.1) / 2.303 = -20 dB, which is sufficient 


		log(bscan, bscanlog);					// switch to logarithmic scale
												//convert to dB = 20 log10(value), from the natural log above
		bscandb = 20.0 * bscanlog / 2.303;

		bscandb.row(4).copyTo(bscandb.row(1));	// masking out the DC in the display
		bscandb.row(4).copyTo(bscandb.row(0));

		//bscandisp=bscandb.rowRange(0, numdisplaypoints);
		tempmat = bscandb.rowRange(0, numdisplaypoints);
		tempmat.copyTo(bscandisp);
		// apply bscanthresholding
		// MatExpr max(const Mat& a, double s)
		//bscandisp = max(bscandisp, bscanthreshold);
		if (clampupper)
		{
			// if this option is selected, set the left upper pixel to 50 dB
			// before normalizing
			bscandisp.at<double>(5, 5) = 50.0;
		}
		normalize(bscandisp, bscandisp, 0, 1, NORM_MINMAX);	// normalize the log plot for display
		bscandisp.convertTo(bscandisp, CV_8UC1, 255.0);


		applyColorMap(bscandisp, cmagI, COLORMAP_JET);
		//putText(cmagI, "^", Point(ascanat - 10, numdisplaypoints), FONT_HERSHEY_COMPLEX, 1, Scalar(255, 255, 255), 3, 8);
		//putText(img,"Text",location, fontface, fonstscale,colorbgr,thickness,linetype, bool bottomLeftOrigin=false);

		imshow("offline Bscan", cmagI);


		if (skeypressed == 1)

		{

			//indexi++;
			sprintf(filename, "bscanoffline%03d", indexi);
			//sprintf(dirname, ".");
			strcpy(dirname, argv[1]);
			savematasbin(pathname, dirname, filename, bscandb);
			savematasimage(pathname, dirname, filename, bscandisp);
			sprintf(filenamec, "bscanofflinec%03d", indexi);
			savematasimage(pathname, dirname, filenamec, cmagI);

			sprintf(textbuffer, "bscanoffline%03d saved.", indexi);
			secrowofstatusimg = Mat::zeros(cv::Size(600, 50), CV_64F);
			putText(statusimg, textbuffer, Point(0, 80), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 3, 1);
			imshow("Status", statusimg);



			skeypressed = 0; // if bscanl is necessary, comment this line, do for bscanl also, then make it 0 



		}// end if skeypressed

		 //bscantransposed = Mat::zeros(Size(numdisplaypoints, oph), CV_64F);



		 //} // end else (if indextemp < averages)



		 /////////////////////////////////////////////////////////

		 //indexi++; 



		key = waitKey(0); // wait indefinitely for keypress

		switch (key)
		{

		case 27: //ESC key
		case 'x':
		case 'X':
			doneflag = 1;
			break;


		case 's':
		case 'S':
		case ' ':

			skeypressed = 1;
			break;

		case 'n':
		case 'N':
			strcpy(pathname, argv[1]);
			sprintf(filename, "/Trig%03d-%03d.png", indexi + 1, 0);
			strcat(pathname, filename);
			if (fexists(pathname))
			{
				indexi++;
				indextemp = 0;
			}
			break;

		case 'p':
		case 'P':
			strcpy(pathname, argv[1]);
			sprintf(filename, "/Trig%03d-%03d.png", indexi - 1, 0);
			strcat(pathname, filename);
			if (fexists(pathname))
			{
				indexi--;
				indextemp = 0;
			}
			break;


		default:
			break;

		}


		if (doneflag == 1)
		{
			break;
		}


	} // while loop end


	return 0;


}

