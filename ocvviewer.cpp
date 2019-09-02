#ifdef _WIN64
#include "stdafx.h"
#include "windows.h"
// anything before a precompiled header is ignored, 
// so no endif here! add #endif to compile on __unix__ !
#endif
#ifdef _WIN64
//#include <qhyccd.h>
#endif


/*
* opencv binary file reader
* has option to save as Matlab readable file.
*

* n for next file, p for previous file
* s to save as
*
* ESC, x or X key quits
*
*
*
* Hari Nandakumar
* 30 Aug 2019  *
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

inline Mat zeropadrowwise(Mat sm, int sn)
{
	// increase fft points sn times 
	// newnumcols = numcols*sn;
	// by fft, zero padding, and then inv fft

	// returns CV_64F

	// guided by https://stackoverflow.com/questions/10269456/inverse-fourier-transformation-in-opencv
	// inspired by Drexler & Fujimoto 2nd ed Section 5.1.10

	// needs fftshift implementation for the zero pad to work correctly if done on borders.
	// or else adding zeros directly to the higher frequencies. 

	// freqcomplex=fftshift(fft(signal));
	// zp2=4*ifft(ifftshift(zpfreqcomplex));

	// result of this way of zero padding in the fourier domain is to resample the same min / max range
	// at a higher sampling rate in the initial domain.
	// So this improves the k linear interpolation.

	Mat origimage;
	Mat fouriertransform, fouriertransformzp;
	Mat inversefouriertransform;

	int numrows = sm.rows;
	int numcols = sm.cols;
	int newnumcols = numcols * sn;

	sm.convertTo(origimage, CV_32F);

	dft(origimage, fouriertransform, DFT_SCALE | DFT_COMPLEX_OUTPUT | DFT_ROWS);

	// implementing fftshift row-wise
	// like https://docs.opencv.org/2.4/doc/tutorials/core/discrete_fourier_transform/discrete_fourier_transform.html
	int cx = fouriertransform.cols / 2;

	// here we assume fouriertransform.cols is even

	Mat LHS(fouriertransform, Rect(0, 0, cx, fouriertransform.rows));   // Create a ROI per half
	Mat RHS(fouriertransform, Rect(cx, 0, cx, fouriertransform.rows)); //  Rect(topleftx, toplefty, w, h), 
																	   // OpenCV typically assumes that the top and left boundary of the rectangle are inclusive, while the right and bottom boundaries are not. 
																	   // https://docs.opencv.org/3.2.0/d2/d44/classcv_1_1Rect__.html

	Mat tmp;                           // swap LHS & RHS
	LHS.copyTo(tmp);
	RHS.copyTo(LHS);
	tmp.copyTo(RHS);

	copyMakeBorder(fouriertransform, fouriertransformzp, 0, 0, floor((newnumcols - numcols) / 2), floor((newnumcols - numcols) / 2), BORDER_CONSTANT, 0.0);
	// this does the zero pad - copyMakeBorder(src, dest, top, bottom, left, right, borderType, value)

	// Now we do the ifftshift before ifft
	cx = fouriertransformzp.cols / 2;
	Mat LHSzp(fouriertransformzp, Rect(0, 0, cx, fouriertransformzp.rows));   // Create a ROI per half
	Mat RHSzp(fouriertransformzp, Rect(cx, 0, cx, fouriertransformzp.rows)); //  Rect(topleftx, toplefty, w, h)

	LHSzp.copyTo(tmp);
	RHSzp.copyTo(LHSzp);
	tmp.copyTo(RHSzp);

	dft(fouriertransformzp, inversefouriertransform, DFT_INVERSE | DFT_REAL_OUTPUT | DFT_ROWS);
	inversefouriertransform.convertTo(inversefouriertransform, CV_64F);

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
	// saves a Mat m by writing to a binary file  f appending .spin, both windows and unix versions
	// p=pathname, d=dirname, f=filename

#ifdef __unix__
	strcpy(p, d);
	strcat(p, "/");
	strcat(p, f);
	strcat(p, ".spin");
	matwrite(p, m);
#else

	strcpy(p, d);
	strcat(p, "\\");		// imwrite needs path with \\ separators, not /, on windows
	strcat(p, f);
	strcat(p, ".spin");
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

int main(int argc, char *argv[])
{ 
	
	int num = 0;
	//qhyccd_handle *camhandle = NULL;
	int ret;
	char id[32];
	//char camtype[16];
	int found = 0;
	unsigned int w, h, bpp = 8, channels, cambitdepth = 16;
	unsigned int offsetx = 0, offsety = 0;
	unsigned int indexi, manualindexi, averages = 1, opw, oph;
	uint  indextemp;
	//uint indextempl;
	uint ascanat = 20;
	uint averagestoggle = 1;


	int camtime = 1, camgain = 1, camspeed = 1, cambinx = 2, cambiny = 2, usbtraffic = 10;
	int camgamma = 1, binvalue = 1, normfactor = 1, normfactorforsave = 25;
	int numfftpoints = 1024;
	int numdisplaypoints = 512;
	bool saveframes = 0;
	bool manualaveraging = 0, saveinterferograms = 0;
	unsigned int manualaverages = 1;
	int movavgn = 0;
	bool clampupper = 0;
	bool ROIreport = 0;

	bool doneflag = 0, skeypressed = 0, bkeypressed = 0, pkeypressed = 0;
	bool jlockin = 0, jkeypressed = 0, ckeypressed = 0;
	bool kkeypressed = 0;
	Mat jmask, jmaskt;
	double lambdamin, lambdamax;
	lambdamin = 816e-9;
	lambdamax = 884e-9;
	int mediann = 5;
	uint increasefftpointsmultiplier = 1;
	double bscanthreshold = -30.0;
	bool rowwisenormalize = 0;
	bool donotnormalize = 1;

	w = 1280;
	h = 960;

	int  fps, key;
	int t_start, t_end;

	
	char dirname[80];
	char filename[20];
	char filenamec[20];
	char pathname[140];
	char lambdamaxstr[40];
	char lambdaminstr[40];
	
	
	
	struct tm *timenow;

	time_t now = time(NULL);
	
	
	
	namedWindow("show", 0); // 0 = WINDOW_NORMAL
	moveWindow("show", 0, 0);

	namedWindow("Bscan", 0); // 0 = WINDOW_NORMAL
	moveWindow("Bscan", 800, 0);

	namedWindow("Status", 0); // 0 = WINDOW_NORMAL
	moveWindow("Status", 0, 500);

	

	/////////////////////////////////////
	// init camera, variables, etc

	cambitdepth = bpp;
	opw = w / binvalue;
	oph = h / binvalue;
	float lambda0 = (lambdamin + lambdamax) / 2;
	float lambdabw = lambdamax - lambdamin;

	unsigned int vertposROI = 10, widthROI = 10;

	Mat data_y(oph, opw, CV_64F);		// the Mat constructor Mat(rows,columns,type)
	Mat data_ylin(oph, numfftpoints, CV_64F);
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

	Mat bscansave0[100];		// allocate buffer to save frames, max 100
	Mat bscansave1[100];		// one buffer is active while other is saved on skeypressed

	Mat jscansave;		// to save j frames

	Mat bscanmanualsave0[100];
	Mat bscanmanualsave1[100];

	Mat interferogramsave0[100];
	Mat interferogramsave1[100];
	Mat interferogrambsave0[100];
	Mat interferogrambsave1[100];
	Mat bscansublog, positivediff;

	bool zeroisactive = 1;

	int nr, nc;

	Mat m, opm, opmvector, bscan, bscanlog, bscandb, bscandisp, bscandispmanual, bscantemp, bscantemp2, bscantemp3, bscantransposed, chan[3];
	Mat tempmat;
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

	resizeWindow("Bscan", oph, numdisplaypoints);		// (width,height)

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
	
	//strcpy(pathname,"bscan001.ocv");
	char const * FilterPatterns[2] =  { "*.ocv", "*.*" };
	char const * OpenFileName = tinyfd_openFileDialog(
		"Open a .OCV file",
		"",
		2,
		FilterPatterns,
		NULL,
		0);

	if (! OpenFileName)
	{
		tinyfd_messageBox(
			"Error",
			"No file chosen. ",
			"ok",
			"error",
			1);
		return 1 ;
	}
	
	mraw = matread(OpenFileName);
	
	minMaxLoc(mraw, &minVal, &maxVal);
	cout << endl << "Min-Max = " << minVal << " to " << maxVal << " dB" << endl;
	
	normalize(mraw, opm, 0.0, 1, NORM_MINMAX);
	
	imshow("show", opm);
	waitKey(0);
	

	/*
	

		doneflag = 0;

		
		t_start = time(NULL);
		fps = 0;

		indexi = 0;
		manualindexi = 0;
		indextemp = 0;
		bscantransposed = Mat::zeros(Size(numdisplaypoints, oph), CV_64F);
		manualaccum = Mat::zeros(Size(oph, numdisplaypoints), CV_64F); // this is transposed version
																	   //bscantransposedl = Mat::zeros(Size(opw/2, oph), CV_64F);

		for (uint p = 0; p<(opw); p++)
		{
			// create modified Bartlett-Hann window
			// https://in.mathworks.com/help/signal/ref/barthannwin.html
			float nn = p;
			float NN = opw - 1;
			barthannwin.at<double>(0, p) = 0.62 - 0.48*std::abs(nn / NN - 0.5) + 0.38*std::cos(2 * pi*(nn / NN - 0.5));

		}
		
		try
		{
			// Retrieve TL device nodemap and print device information
			INodeMap & nodeMapTLDevice = pCam->GetTLDeviceNodeMap();

			//result = PrintDeviceInfo(nodeMapTLDevice);

			// Initialize camera
			pCam->Init();

			// Retrieve GenICam nodemap
			INodeMap & nodeMap = pCam->GetNodeMap();

			////////////////////////////////////////////////
			// Initialize camera settings using Spinnaker SDK
			////////////////////////////////////////////////
			/*
			 in terms of nodes,
				AcquisitionControl -> AcquisitionMode -> Continuous
				AcquisitionControl -> ExposureAuto -> Off
				AcquisitionControl -> ExposureTime -> 1000
				AnalogControl -> GainAuto -> Off
				AnalogControl -> Gain -> 0.0
				AnalogControl -> Gamma -> 1.0
				ImageFormatControl -> BinningSelector -> Sensor
				ImageFormatControl -> BinningHorizontalMode -> Sum
				ImageFormatControl -> BinningVerticalMode -> Sum
				ImageFormatControl -> BinningHorizontal -> cambinx
				ImageFormatControl -> BinningVertical -> cambiny
				* Width becomes halved if cambinx = 2, etc
				* Speed is increased by binning
				ImageFormatControl -> Width -> 1280
				ImageFormatControl -> Height -> 960
				ImageFormatControl -> OffsetX -> 0
				ImageFormatControl -> OffsetY -> 0
				ImageFormatControl -> PixelFormat -> Mono8
				ImageFormatControl -> PixelFormat -> Mono16
				ImageFormatControl -> AdcBitDepth -> Bit12
				BufferHandlingControl -> StreamBufferHandlingMode -> NewestOnly
			*/
			///////////////////////////////////////////////////////////////////////////////////////////////////////
			//Set Buffer Handling
			
			// Only take the latest image, 
			// https://www.flir.com/support-center/iis/machine-vision/application-note/understanding-buffer-handling/
			// https://github.com/RoboCup-SSL/ssl-vision/commit/cd0e342dcbc1ead17bb3c8762f48e5be4e513a32
			/*
			pCam->TLStream.StreamBufferHandlingMode.SetValue(StreamBufferHandlingMode_NewestOnly);
			
			///////////////////////////////////////////////////////////////////////////////////////////////////////
			//Set Acquisition frame rate to a lower value to prevent black frames (due to cpu overload?)
			CBooleanPtr AcquisitionFrameRateEnable = nodeMap.GetNode("AcquisitionFrameRateEnable");
			if (!IsAvailable(AcquisitionFrameRateEnable) || !IsReadable(AcquisitionFrameRateEnable)) {
			  std::cout << "Unable to enable frame rate." << std::endl;
			  return -1;
			}
			AcquisitionFrameRateEnable->SetValue(1);
			
			pCam->AcquisitionFrameRate.SetValue(camspeed);
    
			///////////////////////////////////////////////////////////////////////////////////////////////////////
			//Set Adc bit depth to Bit12
			CEnumerationPtr ptrAdcBitDepth = nodeMap.GetNode("AdcBitDepth");
			if (!IsAvailable(ptrAdcBitDepth) || !IsWritable(ptrAdcBitDepth))
			{
				cout << "Unable to set adc bit depth as ptr is not writable ..." << endl << endl;
				return -1;
			}

			// Retrieve entry node from enumeration node
			CEnumEntryPtr ptrAdcBitDepth12 = ptrAdcBitDepth->GetEntryByName("Bit12");
			if (!IsAvailable(ptrAdcBitDepth12) || !IsReadable(ptrAdcBitDepth12))
			{
				cout << "Unable to set adc bit depth to 12 as entry is not readable..." << endl << endl;
				return -1;
			}

			// Retrieve integer value from entry node
			const int64_t AdcBitDepth12 = ptrAdcBitDepth12->GetValue();

			// Set integer value from entry node as new value of enumeration node
			ptrAdcBitDepth->SetIntValue(AdcBitDepth12);
			
			///////////////////////////////////////////////////////////////////////////////////////////////////////
			//Set auto exposure off
			CEnumerationPtr ptrExposureAuto = nodeMap.GetNode("ExposureAuto");
			if (!IsAvailable(ptrExposureAuto) || !IsWritable(ptrExposureAuto))
			{
				cout << "Unable to set auto exp off as ptr is not writable ..." << endl << endl;
				return -1;
			}

			// Retrieve entry node from enumeration node
			CEnumEntryPtr ptrExposureAutoOff = ptrExposureAuto->GetEntryByName("Off");
			if (!IsAvailable(ptrExposureAutoOff) || !IsReadable(ptrExposureAutoOff))
			{
				cout << "Unable to set auto exp off as entry is not readable..." << endl << endl;
				return -1;
			}

			// Retrieve integer value from entry node
			const int64_t ExposureAutoOff = ptrExposureAutoOff->GetValue();

			// Set integer value from entry node as new value of enumeration node
			ptrExposureAuto->SetIntValue(ExposureAutoOff);
			
			///////////////////////////////////////////////////////////////////////////////////////////////////////
			//Set auto gain off
			CEnumerationPtr ptrGainAuto = nodeMap.GetNode("GainAuto");
			if (!IsAvailable(ptrGainAuto) || !IsWritable(ptrGainAuto))
			{
				cout << "Unable to set auto gain off as ptr is not writable ..." << endl << endl;
				return -1;
			}

			// Retrieve entry node from enumeration node
			CEnumEntryPtr ptrGainAutoOff = ptrGainAuto->GetEntryByName("Off");
			if (!IsAvailable(ptrGainAutoOff) || !IsReadable(ptrGainAutoOff))
			{
				cout << "Unable to set auto gain off as entry is not readable..." << endl << endl;
				return -1;
			}

			// Retrieve integer value from entry node
			const int64_t GainAutoOff = ptrGainAutoOff->GetValue();

			// Set integer value from entry node as new value of enumeration node
			ptrGainAuto->SetIntValue(GainAutoOff);
			
			///////////////////////////////////////////////////////////////////////////////////////////////////////
			//Set gain with QuickSpin
			if (IsReadable(pCam->Gain) && IsWritable(pCam->Gain))
			{
				//pCam->Gain.SetValue(pCam->Gain.GetMin());
				pCam->Gain.SetValue(camgain);

				cout << "Gain set to " << pCam->Gain.GetValue() << " dB ..." << endl;
			}
			else
			{
				cout << "Gain not available..." << endl;
				result = -1;
			}
			// Gain is in dB
			
			///////////////////////////////////////////////////////////////////////////////////////////////////////
			//Set exp with QuickSpin
			if (IsReadable(pCam->ExposureTime) && IsWritable(pCam->ExposureTime))
			{
				pCam->ExposureTime.SetValue(camtime);

				cout << "Exp set to " << pCam->ExposureTime.GetValue() << " microsec ..." << endl;
			}
			else
			{
				cout << "Exp not available..." << endl;
				result = -1;
			}
			
			///////////////////////////////////////////////////////////////////////////////////////////////////////
			//Set offsetx 
			if (IsReadable(pCam->OffsetX) && IsWritable(pCam->OffsetX))
			{
				//pCam->OffsetX.SetValue(pCam->OffsetX.GetMin());
				pCam->OffsetX.SetValue(offsetx);

				
			}
			else
			{
				cout << "Offset X not available..." << endl;
				result = -1;
			}
			
			///////////////////////////////////////////////////////////////////////////////////////////////////////
			//Set offsety 
			if (IsReadable(pCam->OffsetY) && IsWritable(pCam->OffsetY))
			{
				pCam->OffsetY.SetValue(offsety);

				
			}
			else
			{
				cout << "Offset Y not available..." << endl;
				result = -1;
			}
			
			///////////////////////////////////////////////////////////////////////////////////////////////////////
			//Set Width 
			if (IsReadable(pCam->Width) && IsWritable(pCam->Width))
			{
				pCam->Width.SetValue(w);

				
			}
			else
			{
				cout << "Width not available..." << endl;
				result = -1;
			}
			
			///////////////////////////////////////////////////////////////////////////////////////////////////////
			//Set Height 
			if (IsReadable(pCam->Height) && IsWritable(pCam->Height))
			{
				pCam->Height.SetValue(h);

				
			}
			else
			{
				cout << "Height not available..." << endl;
				result = -1;
			}
			
			///////////////////////////////////////////////////////////////////////////////////////////////////////
			//Set PixelFormat - bitdepth 
			if (IsReadable(pCam->PixelFormat) && IsWritable(pCam->PixelFormat))
			{
				if (cambitdepth == 8)
				pCam->PixelFormat.SetValue(PixelFormat_Mono8);
				else
				pCam->PixelFormat.SetValue(PixelFormat_Mono16);

				cout << "Pixel format set to " << pCam->PixelFormat.GetCurrentEntry()->GetSymbolic() << "..." << endl;
			}
			else
			{
				cout << "Pixel format not available..." << endl;
				result = -1;
			}
			
			///////////////////////////////////////////////////////////////////////////////////////////////////////
			//Set Gamma 
			if (IsReadable(pCam->Gamma) && IsWritable(pCam->Gamma))
			{
				pCam->Gamma.SetValue(camgamma);
				
				cout << "Gamma set to " << pCam->Gamma.GetValue() << "..." << endl;
			}
			else
			{
				cout << "Gamma not available..." << endl;
				result = -1;
			}
			
			///////////////////////////////////////////////////////////////////////////////////////////////////////
			//Set BinningSelector
			CEnumerationPtr ptrBinningSelector = nodeMap.GetNode("BinningSelector");
			if (!IsAvailable(ptrBinningSelector) || !IsWritable(ptrBinningSelector))
			{
				cout << "Unable to set binning selector as ptr is not writable ..." << endl << endl;
				return -1;
			}

			// Retrieve entry node from enumeration node
			CEnumEntryPtr ptrBinningSelectorSensor = ptrBinningSelector->GetEntryByName("All");
			if (!IsAvailable(ptrBinningSelectorSensor) || !IsReadable(ptrBinningSelectorSensor))
			{
				cout << "Unable to set binning selector to all as entry is not readable..." << endl << endl;
				return -1;
			}

			// Retrieve integer value from entry node
			const int64_t BinningSelectorSensor = ptrBinningSelectorSensor->GetValue();

			// Set integer value from entry node as new value of enumeration node
			ptrBinningSelector->SetIntValue(BinningSelectorSensor);
			
			///////////////////////////////////////////////////////////////////////////////////////////////////////
			//Set BinningHorizontalMode to Sum and not Average
			CEnumerationPtr ptrBinningHorizontalMode = nodeMap.GetNode("BinningHorizontalMode");
			if (!IsAvailable(ptrBinningHorizontalMode) || !IsWritable(ptrBinningHorizontalMode))
			{
				cout << "Unable to set binning h mode as ptr is not writable ..." << endl << endl;
				return -1;
			}

			// Retrieve entry node from enumeration node
			CEnumEntryPtr ptrBinningHorizontalModeSum = ptrBinningHorizontalMode->GetEntryByName("Sum");
			if (!IsAvailable(ptrBinningHorizontalModeSum) || !IsReadable(ptrBinningHorizontalModeSum))
			{
				cout << "Unable to set binning h mode to sum as entry is not readable..." << endl << endl;
				return -1;
			}

			// Retrieve integer value from entry node
			const int64_t BinningHorizontalModeSum = ptrBinningHorizontalModeSum->GetValue();

			// Set integer value from entry node as new value of enumeration node
			ptrBinningHorizontalMode->SetIntValue(BinningHorizontalModeSum);
			
			///////////////////////////////////////////////////////////////////////////////////////////////////////
			//Set BinningVerticalMode to Sum and not Average
			CEnumerationPtr ptrBinningVerticalMode = nodeMap.GetNode("BinningVerticalMode");
			if (!IsAvailable(ptrBinningVerticalMode) || !IsWritable(ptrBinningVerticalMode))
			{
				cout << "Unable to set binning h mode as ptr is not writable ..." << endl << endl;
				return -1;
			}

			// Retrieve entry node from enumeration node
			CEnumEntryPtr ptrBinningVerticalModeSum = ptrBinningVerticalMode->GetEntryByName("Sum");
			if (!IsAvailable(ptrBinningVerticalModeSum) || !IsReadable(ptrBinningVerticalModeSum))
			{
				cout << "Unable to set binning h mode to sum as entry is not readable..." << endl << endl;
				return -1;
			}

			// Retrieve integer value from entry node
			const int64_t BinningVerticalModeSum = ptrBinningVerticalModeSum->GetValue();

			// Set integer value from entry node as new value of enumeration node
			ptrBinningVerticalMode->SetIntValue(BinningVerticalModeSum);
			
			///////////////////////////////////////////////////////////////////////////////////////////////////////
			//Set cambinx 
			if (IsReadable(pCam->BinningHorizontal) && IsWritable(pCam->BinningHorizontal))
			{
				pCam->BinningHorizontal.SetValue(cambinx);
				
				cout << "BinningHorizontal set to " << pCam->BinningHorizontal.GetValue() << "..." << endl;
			}
			else
			{
				cout << "BinningHorizontal not available..." << endl;
				result = -1;
			}
			
			///////////////////////////////////////////////////////////////////////////////////////////////////////
			//Set cambiny 
			if (IsReadable(pCam->BinningVertical) && IsWritable(pCam->BinningVertical))
			{
				pCam->BinningVertical.SetValue(cambiny);
				
				cout << "BinningVertical set to " << pCam->BinningVertical.GetValue() << "..." << endl;
			}
			else
			{
				cout << "BinningVertical not available..." << endl;
				result = -1;
			}

		
			///////////////////////////////////////////////////////////////////////////////////////////////////////
			//Changing to continuous acq
			CEnumerationPtr ptrAcquisitionMode = nodeMap.GetNode("AcquisitionMode");
			if (!IsAvailable(ptrAcquisitionMode) || !IsWritable(ptrAcquisitionMode))
			{
				cout << "Unable to set continuous acquisition mode as ptr is not writable ..." << endl << endl;
				return -1;
			}

			// Retrieve entry node from enumeration node
			CEnumEntryPtr ptrAcquisitionModeContinuous = ptrAcquisitionMode->GetEntryByName("Continuous");
			if (!IsAvailable(ptrAcquisitionModeContinuous) || !IsReadable(ptrAcquisitionModeContinuous))
			{
				cout << "Unable to set continuous acquisition mode as entry is not readable..." << endl << endl;
				return -1;
			}

			// Retrieve integer value from entry node
			const int64_t acquisitionModeContinuous = ptrAcquisitionModeContinuous->GetValue();

			// Set integer value from entry node as new value of enumeration node
			ptrAcquisitionMode->SetIntValue(acquisitionModeContinuous);
			/////////////////////////////////////////////////////////////////////////////////////////////////////////
			
			
			ImagePtr pResultImage;
			ImagePtr convertedImage;
			// trying begin and end acq, to see if it will improve fps
			// this needs StreamBufferHandlingMode -> NewestOnly
			pCam->BeginAcquisition();
			//////////////////

		while (1)		//camera frames acquisition loop, which is inside the try
		{
			//ret = GetQHYCCDLiveFrame(camhandle, &w, &h, &bpp, &channels, mraw.data);
			
			ret = 0;
			
			while(ret==0)
			{
				pResultImage = pCam->GetNextImage();
				
				if(pResultImage->IsIncomplete())
				{
					ret=0;
				}
				else
				{
					ret=1;
					convertedImage = pResultImage;
					if (bpp == 8)
					{
						// Mono8 w x h 
						
						mraw = cv::Mat(h, w, CV_8UC1, convertedImage->GetData(), convertedImage->GetStride());
					}
					else
					{
						//Mono16 w x h
						mraw = cv::Mat(h, w, CV_16UC1, convertedImage->GetData(), convertedImage->GetStride());
					}

				}
			} // end of while IsIncomplete
			
			// pResultImage has to be released to avoid buffer filling up
			pResultImage->Release();
			
			if (ret == 1)
			{
				
				
				
				//median filter while the numbers are still int
				if (mediann>0)
					medianBlur(mraw, m, mediann);
				else
					mraw.copyTo(m);
				 
					resize(m, opm, Size(), 1.0 / binvalue, 1.0 / binvalue, INTER_AREA);	// binning (averaging)
					// opencv seems to automatically display CV_16U correctly
				imshow("show", opm);

				if (saveinterferograms)
				{
					// save mraw to active buffer
					// inactive buffer is saved to disk when skeypressed
					if (zeroisactive)
					{
						mraw.copyTo(interferogramsave0[indextemp]);
						opm.copyTo(interferogrambsave0[indextemp]);
						//printf("Saved to interferogramsave0[%d]\n",indextemp);
						//sprintf(debugwinname,"debug%d",baccumcount);
						//imshow(debugwinname,interferogramsave0[indextemp]);
						//waitKey(30);
					}
					else
					{
						mraw.copyTo(interferogramsave1[indextemp]);
						opm.copyTo(interferogrambsave1[indextemp]);
						//printf("Saved to interferogramsave1[%d]\n",indextemp);
						//sprintf(debugwinname,"debug%d",baccumcount);
						//imshow(debugwinname,interferogramsave1[indextemp]);
						//waitKey(30);
					}

				}


				opm.convertTo(data_y, CV_64F);	// initialize data_y

												// smoothing by weighted moving average
				if (movavgn > 0)
					data_y = smoothmovavg(data_y, movavgn);



				//transpose(opm, data_y); 		// void transpose(InputArray src, OutputArray dst)
				// because we actually want the columns and not rows
				// using DFT_ROWS
				// But that has rolling shutter issues, so going back to rows

				if (bkeypressed == 1)

				{
					if (saveinterferograms)
					{
						// in this case, formerly active buffer is saved to disk when bkeypressed
						// since no further 
						// and all accumulation is done
						Mat activeMat, activeMatb, activeMat64;
						for (uint ii = 0; ii<averagestoggle; ii++)
						{
							if (zeroisactive)
							{
								activeMat = interferogramsave1[ii];
								activeMatb = interferogrambsave1[ii];
							}
							else
							{
								activeMat = interferogramsave0[ii];
								activeMatb = interferogrambsave0[ii];
							}

							sprintf(filename, "rawframeb%03d-%03d", indexi, ii);
							savematasimage(pathname, dirname, filename, activeMat);
							activeMatb.convertTo(activeMat64, CV_64F);
							accumulate(activeMat64, baccum);
						}
						baccum.copyTo(data_yb);		// saves the "background" or source spectrum
						if (rowwisenormalize)
							normalizerows(data_yb, data_yb, 0.0001, 1);
						if (!donotnormalize)
							normalize(data_yb, data_yb, 0.0001, 1, NORM_MINMAX);
						else
							data_yb = data_yb / averagestoggle;

						bkeypressed = 0;

					}

					else
					{
						if (baccumcount < averagestoggle)
						{
							accumulate(data_y, baccum);
							// save the raw frame to buffer

							baccumcount++;
						}
						else
						{
							baccum.copyTo(data_yb);		// saves the "background" or source spectrum

							if (rowwisenormalize)
								normalizerows(data_yb, data_yb, 0.0001, 1);
							if (!donotnormalize)
								normalize(data_yb, data_yb, 0.0001, 1, NORM_MINMAX);
							else
								data_yb = data_yb / averagestoggle;
							bkeypressed = 0;
							baccumcount = 0;

						}
					} // end if not saveinterferograms

					sprintf(textbuffer, "S(k) saved.");
					secrowofstatusimg = Mat::zeros(cv::Size(600, 50), CV_64F);
					putText(statusimg, textbuffer, Point(0, 80), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 3, 1);
					imshow("Status", statusimg);
					resizeWindow("Status", 600, 300);

					if (manualaveraging)
					{
						averagestoggle = 1;
					}

				}

				if (pkeypressed == 1)

				{

					data_y.copyTo(data_yp);		// saves the pi shifted or J0 spectrum	
					if (saveinterferograms)
					{
						// only a single frame to be saved when pkeypressed
						Mat temp;
						data_y.convertTo(temp, CV_8UC1);
						sprintf(filename, "rawframepbin%03d", indexi);
						savematasimage(pathname, dirname, filename, temp);
						sprintf(filename, "rawframep%03d", indexi);
						savematasimage(pathname, dirname, filename, mraw);
					}
					data_yp.convertTo(data_yp, CV_64F);
					if (rowwisenormalize)
						normalizerows(data_yp, data_yp, 0, 1);
					if (!donotnormalize)
						normalize(data_yp, data_yp, 0, 1, NORM_MINMAX);
					pkeypressed = 0;

				}
				fps++;
				t_end = time(NULL);
				if (t_end - t_start >= 5)
				{
					//printf("fps = %d\n", fps / 5);
					opm.copyTo(opmvector);
					opmvector.reshape(0, 1);	//make it into a row array
					minMaxLoc(opmvector, &minVal, &maxVal);
					sprintf(textbuffer, "fps = %d  Max intensity = %d", fps / 5, int(floor(maxVal)));
					firstrowofstatusimg = Mat::zeros(cv::Size(600, 50), CV_64F);
					putText(statusimg, textbuffer, Point(0, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 3, 1);
					sprintf(textbuffer, "%03d images acq.", indextemp);
					secrowofstatusimgRHS = Mat::zeros(cv::Size(300, 50), CV_64F);
					putText(statusimg, textbuffer, Point(300, 80), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 3, 1);
					imshow("Status", statusimg);
					if (ROIreport)
						printMinMaxAscan(bscandb, ascanat, numdisplaypoints, statusimg);
					fps = 0;
					t_start = time(NULL);
				}

				////////////////////////////////////////////

				// apodize 
				// data_y = ( (data_y - data_yb) ./ data_yb ).*gausswin
				data_y.convertTo(data_y, CV_64F);
				if (rowwisenormalize)
					normalizerows(data_y, data_y, 0, 1);
				if (!donotnormalize)
					normalize(data_y, data_y, 0, 1, NORM_MINMAX);
				//data_yb.convertTo(data_yb, CV_64F);
				//
				data_y = (data_y - data_yp) / data_yb;


				for (int p = 0; p<(data_y.rows); p++)
				{
					//DC removal
					Scalar meanval = mean(data_y.row(p));
					data_y.row(p) = data_y.row(p) - meanval(0);		// Only the first value of the scalar is useful for us

																	//windowing
					multiply(data_y.row(p), barthannwin, data_y.row(p));
				}

				//increasing number of points by zero padding
				if (increasefftpointsmultiplier > 1)
					data_y = zeropadrowwise(data_y, increasefftpointsmultiplier);


				// interpolate to linear k space
				for (int p = 0; p<(data_y.rows); p++)
				{
					for (int q = 1; q<(data_y.cols); q++)
					{
						//find the slope of the data_y at each of the non-linear ks
						slopes.at<double>(p, q) = data_y.at<double>(p, q) - data_y.at<double>(p, q - 1);
						// in the .at notation, it is <double>(y,x)
						//printf("slopes(%d,%d)=%f \n",p,q,slopes.at<double>(p,q));
					}
					// initialize the first slope separately
					slopes.at<double>(p, 0) = slopes.at<double>(p, 1);


					for (int q = 1; q<(data_ylin.cols - 1); q++)
					{
						//find the value of the data_ylin at each of the klinear points
						// data_ylin = data_y(nearestkindex) + fractionalk(nearestkindex)*slopes(nearestkindex)
						//std::cout << "q=" << q << " nearestk=" << nearestkindex.at<int>(0,q) << std::endl;
						data_ylin.at<double>(p, q) = data_y.at<double>(p, nearestkindex.at<int>(0, q))
							+ fractionalk.at<double>(nearestkindex.at<int>(0, q))
							* slopes.at<double>(p, nearestkindex.at<int>(0, q));
						//printf("data_ylin(%d,%d)=%f \n",p,q,data_ylin.at<double>(p,q));
					}
					//data_ylin.at<double>(p, 0) = 0;
					//data_ylin.at<double>(p, numfftpoints) = 0;

				}

				// InvFFT

				Mat planes[] = { Mat_<float>(data_ylin), Mat::zeros(data_ylin.size(), CV_32F) };
				Mat complexI;
				merge(planes, 2, complexI);         // Add to the expanded another plane with zeros

				dft(complexI, complexI, DFT_ROWS | DFT_INVERSE);            // this way the result may fit in the source matrix

																			// compute the magnitude and switch to logarithmic scale
																			// => log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))
				split(complexI, planes);                   // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
				magnitude(planes[0], planes[1], magI);


				if (indextemp < averagestoggle)
				{
					bscantemp = magI.colRange(0, numdisplaypoints);
					bscantemp.convertTo(bscantemp, CV_64F);
					accumulate(bscantemp, bscantransposed);
					if (saveframes == 1)
					{
						// save the individual frames before averaging also
						if (zeroisactive)
							bscantemp.copyTo(bscansave0[indextemp]);
						else
							bscantemp.copyTo(bscansave1[indextemp]);
					}

					indextemp++;

				}

				if (indextemp >= averagestoggle)
				{
					sprintf(textbuffer, "%03d images acq.", indextemp);
					secrowofstatusimgRHS = Mat::zeros(cv::Size(300, 50), CV_64F);
					putText(statusimg, textbuffer, Point(300, 80), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 3, 1);
					imshow("Status", statusimg);
					indextemp = 0;
					// we will also toggle the buffers, at the end of this 'else' code block

					transpose(bscantransposed, bscan);
					bscan = bscan / averagestoggle;
					bscan += Scalar::all(0.00001);   	// to prevent log of 0  
														// 20.0 * log(0.1) / 2.303 = -20 dB, which is sufficient 

					if (jlockin)
					{
						Mat jdiff = bscan - jscansave;	// these are in linear scale
						jdiff.copyTo(positivediff);		// just to initialize the Mat
						makeonlypositive(jdiff, positivediff);
						positivediff += 0.001;			// to avoid log(0)

					}


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
					bscandisp = max(bscandisp, bscanthreshold);
					if (clampupper)
					{
						// if this option is selected, set the left upper pixel to 50 dB
						// before normalizing
						bscandisp.at<double>(5, 5) = 50.0;
					}
					normalize(bscandisp, bscandisp, 0, 1, NORM_MINMAX);	// normalize the log plot for display
					bscandisp.convertTo(bscandisp, CV_8UC1, 255.0);

					if (jlockin)
					{
						// code to display and save subtracted frame
						log(positivediff, bscansublog);
						bscandispmanual = 20.0 * bscansublog / 2.303;

						// apply bscanthresholding
						bscandispmanual = max(bscandispmanual, bscanthreshold);

						normalize(bscandispmanual, bscandispmanual, 0, 1, NORM_MINMAX);	// normalize the log plot for display
						bscandispmanual.convertTo(bscandispmanual, CV_8UC1, 255.0);
						applyColorMap(bscandispmanual, cmagImanual, COLORMAP_JET);

						imshow("Bscan subtracted", cmagImanual);


						// and save - similar code as in skeypressed
						//////////////////////////////////////////
						manualindexi++;
						sprintf(filename, "bscansub%03d", manualindexi);
						sprintf(filenamec, "bscansubc%03d", manualindexi);
						savematasbin(pathname, dirname, filename, manualaccum);
						savematasdata(outfile, filename, manualaccum);
						savematasimage(pathname, dirname, filename, bscandispmanual);
						savematasimage(pathname, dirname, filenamec, cmagImanual);

					}

					applyColorMap(bscandisp, cmagI, COLORMAP_JET);
					putText(cmagI, "^", Point(ascanat - 10, numdisplaypoints), FONT_HERSHEY_COMPLEX, 1, Scalar(255, 255, 255), 3, 8);
					//putText(img,"Text",location, fontface, fonstscale,colorbgr,thickness,linetype, bool bottomLeftOrigin=false);

					imshow("Bscan", cmagI);
					if (ROIreport)
						printAvgROI(bscandb, ascanat, vertposROI, widthROI, ROIplot, ROIploti, statusimg);

					if (jkeypressed == 1)
					{
						bscan.copyTo(jscansave);
						jlockin = 1;			// setting the boolean flag
						jkeypressed = 0;
					}

					if (ckeypressed == 1)
					{
						// clear the thresholding boolean
						jlockin = 0;
						ckeypressed = 0;
					}


					if (skeypressed == 1)

					{

						indexi++;
						sprintf(filename, "bscan%03d", indexi);
						savematasdata(outfile, filename, bscandb);
						savematasbin(pathname, dirname, filename, bscandb);
						savematasimage(pathname, dirname, filename, bscandisp);
						sprintf(filenamec, "bscanc%03d", indexi);
						savematasimage(pathname, dirname, filenamec, cmagI);

						sprintf(textbuffer, "bscan%03d saved.", indexi);
						secrowofstatusimg = Mat::zeros(cv::Size(600, 50), CV_64F);
						putText(statusimg, textbuffer, Point(0, 80), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 3, 1);
						imshow("Status", statusimg);

						if (jlockin)
						{
							// save the respective j image also
							sprintf(filename, "jscan%03d", indexi);
							savematasbin(pathname, dirname, filename, jscansave);
							savematasdata(outfile, filename, jscansave);
							savematasimage(pathname, dirname, filename, jscansave);

						}
						if (saveinterferograms)
						{
							//sprintf(filename, "linearized%03d", indexi);
							//savematasdata(outfile, filename, data_ylin);
							//normalize(data_ylin, bscantemp2, 0, 255, NORM_MINMAX);	// normalize the log plot for save
							//bscantemp2.convertTo(bscantemp2, CV_8UC1, 1.0);		// imwrite needs 0-255 CV_8U
							//savematasimage(pathname, dirname, filename, bscantemp2);
							// in this case, formerly active buffer is saved to disk when bkeypressed
							// since no further 
							// and all accumulation is done
							Mat activeMat;
							for (uint ii = 0; ii<averagestoggle; ii++)
							{
								if (zeroisactive)
								{
									activeMat = interferogramsave1[ii];
								}
								else
								{
									activeMat = interferogramsave0[ii];
								}

								sprintf(filename, "rawframe%03d-%03d", indexi, ii);
								savematasimage(pathname, dirname, filename, activeMat);

							}

						}

						if (saveframes == 1)
						{
							for (uint ii = 0; ii<averagestoggle; ii++)
							{
								// save the bscansave array after processing
								if (zeroisactive)
									transpose(bscansave1[ii], bscantemp2);	// don't touch the active buffer
								else
									transpose(bscansave0[ii], bscantemp2);
								bscantemp2 += Scalar::all(0.000001);   	// to prevent log of 0                 
								log(bscantemp2, bscantemp2);					// switch to logarithmic scale
																				//convert to dB = 20 log10(value), from the natural log above
								bscantemp2 = 20.0 * bscantemp2 / 2.303;
								normalize(bscantemp2, bscantemp2, 0, 1, NORM_MINMAX);	// normalize the log plot for save
								bscantemp2.convertTo(bscantemp2, CV_8UC1, 255.0);		// imwrite needs 0-255 CV_8U
								sprintf(filename, "bscan%03d-%03d", indexi, ii);
								savematasimage(pathname, dirname, filename, bscantemp2);

								if (saveinterferograms)
								{
									// inactive buffer is saved to disk when skeypressed
									// and all accumulation is done
									for (uint ii = 0; ii<averagestoggle; ii++)
									{
										sprintf(filename, "rawframe%03d-%03d", indexi, ii);
										if (zeroisactive)
											savematasimage(pathname, dirname, filename, interferogramsave1[ii]);
										else
											savematasimage(pathname, dirname, filename, interferogramsave0[ii]);
									}

								}


							}
						}

						skeypressed = 0; // if bscanl is necessary, comment this line, do for bscanl also, then make it 0 

						if (manualaveraging)
						{
							if (manualaccumcount < manualaverages)
							{
								accumulate(bscan, manualaccum);

								if (saveframes == 1)
								{
									// save the individual frames before averaging also
									if (zeroisactive)
										bscan.copyTo(bscanmanualsave0[manualaccumcount]);
									else
										bscan.copyTo(bscanmanualsave1[manualaccumcount]);
								}

								manualaccumcount++;
							}
							else
							{
								manualaccumcount = 0;
								manualaccum = manualaccum / manualaverages;
								//printf("depth of manualaccum is %d \n", manualaccum.depth());
								log(manualaccum, manualaccum);					// switch to logarithmic scale
																				//convert to dB = 20 log10(value), from the natural log above
								bscandispmanual = 20.0 * manualaccum / 2.303;

								// apply bscanthresholding
								bscandispmanual = max(bscandispmanual, bscanthreshold);

								normalize(bscandispmanual, bscandispmanual, 0, 1, NORM_MINMAX);	// normalize the log plot for display
								bscandispmanual.convertTo(bscandispmanual, CV_8UC1, 255.0);
								applyColorMap(bscandispmanual, cmagImanual, COLORMAP_JET);

								imshow("Bscanm", cmagImanual);


								// and save - similar code as in skeypressed
								//////////////////////////////////////////
								manualindexi++;
								sprintf(filename, "bscanman%03d", manualindexi);
								sprintf(filenamec, "bscanmanc%03d", manualindexi);
								savematasdata(outfile, filename, manualaccum);
								savematasbin(pathname, dirname, filename, manualaccum);
								savematasimage(pathname, dirname, filename, bscandispmanual);
								savematasimage(pathname, dirname, filenamec, cmagImanual);

								manualaccum = Mat::zeros(Size(oph, numdisplaypoints), CV_64F);


								if (saveframes == 1)
								{
									for (uint ii = 0; ii<manualaverages; ii++)
									{
										// save the bscanmanualsave array after processing
										//transpose(bscanmanualsave[ii], bscantemp2); - is already transposed
										if (zeroisactive)
											bscanmanualsave1[ii].copyTo(bscantemp3);		// save only the inactive buffer
										else
											bscanmanualsave0[ii].copyTo(bscantemp3);

										bscantemp3.convertTo(bscantemp3, CV_64F);
										bscantemp3 += Scalar::all(0.000001);   	// to prevent log of 0       

										log(bscantemp3, bscantemp3);					// switch to logarithmic scale
																						//convert to dB = 20 log10(value), from the natural log above
										bscantemp3 = 20.0 * bscantemp3 / 2.303;
										normalize(bscantemp3, bscantemp3, 0, 1, NORM_MINMAX);	// normalize the log plot for save
										bscantemp3.convertTo(bscantemp3, CV_8UC1, 255.0);		// imwrite needs 0-255 CV_8U
										sprintf(filename, "bscanm%03d-%03d", manualindexi, ii);
										savematasimage(pathname, dirname, filename, bscantemp3);

									}



								} // end if saveframes

							}  //////////////end code to save manual////////////

						} // end if manual averaging


					} // end if skeypressed

					bscantransposed = Mat::zeros(Size(numdisplaypoints, oph), CV_64F);

					// toggle the buffers
					if (zeroisactive)
						zeroisactive = 0;
					else
						zeroisactive = 1;

				} // end else (if indextemp < averages)

				  
				  ////////////////////////////////////////////


				key = waitKey(0); // wait indefinitely for keypress
								  
								  
				bool expchanged = 0;

				switch (key)
				{

				case 27: //ESC key
				case 'x':
				case 'X':
					doneflag = 1;
					break;

				case '+':
				case '=':

					camtime = camtime + 100;
					expchanged = 1;
					
					break;

				case '-':
				case '_':

					camtime = camtime - 100;
					if (camtime < 8)	// spinnaker has a min of 8 microsec
						camtime = 8;
					expchanged = 1;
					
					break;

				case 'U':

					camtime = camtime + 10000;
					expchanged = 1;
					break;
					
				case 'D':

					camtime = camtime - 10000;
					if (camtime < 8)
						camtime = 8;
					expchanged = 1;
					
					break;
					
				case 'u':

					camtime = camtime + 1000;
					expchanged = 1;
					
					break;
					
				case 'd':

					camtime = camtime - 1000;
					if (camtime < 8)
						camtime = 8;
					expchanged = 1;
					
					break;
					

				case 's':
				case 'S':
				case ' ':

					skeypressed = 1;
					break;

				case 'b':
				case 'B':

					bkeypressed = 1;
					break;

				case 'p':
				case 'P':

					pkeypressed = 1;
					break;

				case 'j':
				case 'J':

					jkeypressed = 1;
					break;
					
				case 'k':
				case 'K':

					kkeypressed = 1;
					break;

				case 'c':
				case 'C':

					ckeypressed = 1;
					break;

				case ']':

					bscanthreshold += 1.0;
					sprintf(textbuffer, "bscanthreshold = %f", bscanthreshold);
					secrowofstatusimg = Mat::zeros(cv::Size(600, 50), CV_64F);
					putText(statusimg, textbuffer, Point(0, 80), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 3, 1);
					imshow("Status", statusimg);
					break;

				case '[':

					bscanthreshold -= 1.0;
					sprintf(textbuffer, "bscanthreshold = %f", bscanthreshold);
					secrowofstatusimg = Mat::zeros(cv::Size(600, 50), CV_64F);
					putText(statusimg, textbuffer, Point(0, 80), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 3, 1);
					imshow("Status", statusimg);
					break;

				case '(':
					if (ascanat > 10)
						ascanat -= 10;

					sprintf(textbuffer, "ascanat = %d", ascanat);
					secrowofstatusimg = Mat::zeros(cv::Size(600, 50), CV_64F);
					putText(statusimg, textbuffer, Point(0, 80), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 3, 1);
					imshow("Status", statusimg);
					if (ROIreport)
						printMinMaxAscan(bscandb, ascanat, numdisplaypoints, statusimg);
					break;

				case '9':
					if (ascanat > 0)
						ascanat -= 1;

					sprintf(textbuffer, "ascanat = %d", ascanat);
					secrowofstatusimg = Mat::zeros(cv::Size(600, 50), CV_64F);
					putText(statusimg, textbuffer, Point(0, 80), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 3, 1);
					imshow("Status", statusimg);
					if (ROIreport)
						printMinMaxAscan(bscandb, ascanat, numdisplaypoints, statusimg);
					break;

				case ')':
					if (ascanat < (oph - 11))
						ascanat += 10;

					sprintf(textbuffer, "ascanat = %d", ascanat);
					secrowofstatusimg = Mat::zeros(cv::Size(600, 50), CV_64F);
					putText(statusimg, textbuffer, Point(0, 80), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 3, 1);
					imshow("Status", statusimg);
					if (ROIreport)
						printMinMaxAscan(bscandb, ascanat, numdisplaypoints, statusimg);
					break;
				case '0':
					if (ascanat < (oph - 1))
						ascanat += 1;

					sprintf(textbuffer, "ascanat = %d", ascanat);
					secrowofstatusimg = Mat::zeros(cv::Size(600, 50), CV_64F);
					putText(statusimg, textbuffer, Point(0, 80), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 3, 1);
					imshow("Status", statusimg);
					if (ROIreport)
						printMinMaxAscan(bscandb, ascanat, numdisplaypoints, statusimg);
					break;

				case 'W':
					if ((ascanat + widthROI) < (oph - 1))
						widthROI += 1;

					sprintf(textbuffer, "ROI width = %d", widthROI);
					secrowofstatusimg = Mat::zeros(cv::Size(600, 50), CV_64F);
					putText(statusimg, textbuffer, Point(0, 80), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 3, 1);
					imshow("Status", statusimg);
					if (ROIreport)
						printAvgROI(bscandb, ascanat, vertposROI, widthROI, ROIplot, ROIploti, statusimg);
					break;

				case 'w':
					if (widthROI > 2)
						widthROI -= 1;

					sprintf(textbuffer, "ROI width = %d", widthROI);
					secrowofstatusimg = Mat::zeros(cv::Size(600, 50), CV_64F);
					putText(statusimg, textbuffer, Point(0, 80), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 3, 1);
					imshow("Status", statusimg);
					if (ROIreport)
						printAvgROI(bscandb, ascanat, vertposROI, widthROI, ROIplot, ROIploti, statusimg);
					break;

				case 'h':
					if (vertposROI < (numdisplaypoints - 1))
						vertposROI += 1;

					sprintf(textbuffer, "ROI vertical position = %d ", vertposROI);
					secrowofstatusimg = Mat::zeros(cv::Size(600, 50), CV_64F);
					putText(statusimg, textbuffer, Point(0, 80), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 3, 1);
					imshow("Status", statusimg);
					if (ROIreport)
						printAvgROI(bscandb, ascanat, vertposROI, widthROI, ROIplot, ROIploti, statusimg);
					break;

				case 'H':
					if (vertposROI > 2)
						vertposROI -= 1;

					sprintf(textbuffer, "ROI vertical position = %d", vertposROI);
					secrowofstatusimg = Mat::zeros(cv::Size(600, 50), CV_64F);
					putText(statusimg, textbuffer, Point(0, 80), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 3, 1);
					imshow("Status", statusimg);
					if (ROIreport)
						printAvgROI(bscandb, ascanat, vertposROI, widthROI, ROIplot, ROIploti, statusimg);
					break;

				case 'a':
				case 'A':
					if (averagestoggle == 1)
						averagestoggle = averages;
					else
						averagestoggle = 1;
					sprintf(textbuffer, "Now averaging %d bscans.", averagestoggle);
					secrowofstatusimg = Mat::zeros(cv::Size(600, 50), CV_64F);
					putText(statusimg, textbuffer, Point(0, 80), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 3, 1);
					imshow("Status", statusimg);
					break;

				case 'e':
				case 'E':
					if (ROIreport == 1)
					{
						ROIreport = 0;
						sprintf(textbuffer, "Supressing rEporting/plotting ROI averages.");
						secrowofstatusimg = Mat::zeros(cv::Size(600, 50), CV_64F);
						putText(statusimg, textbuffer, Point(0, 80), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 3, 1);
						imshow("Status", statusimg);
					}
					else
					{
						ROIreport = 1;
						sprintf(textbuffer, "Now rEporting/plotting ROI averages.");
						secrowofstatusimg = Mat::zeros(cv::Size(600, 50), CV_64F);
						putText(statusimg, textbuffer, Point(0, 80), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 3, 1);
						imshow("Status", statusimg);
						//printAvgROI(bscandb, ascanat, vertposROI, widthROI, ROIplot, ROIploti, statusimg);
						//moveWindow("ROI intensity", 800, 400);
					}
					break;

				case 'q':
				case 'Q':
					if (clampupper == 1)
						clampupper = 0;
					else
						clampupper = 1;
					break;

				default:
					break;

				}
				
				if (expchanged == 1)
				{
				//Set exp with QuickSpin
					ret = 0;
					if (IsReadable(pCam->ExposureTime) && IsWritable(pCam->ExposureTime))
					{
						pCam->ExposureTime.SetValue(camtime);
						ret = 1;
					}
					if (ret == 1)
					{
						sprintf(textbuffer, "Exp time = %d ", camtime);
						secrowofstatusimg = Mat::zeros(cv::Size(600, 50), CV_64F);
						putText(statusimg, textbuffer, Point(0, 80), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 3, 1);
						imshow("Status", statusimg);

					}
					else
					{
						sprintf(textbuffer, "CONTROL_EXPOSURE failed");
						secrowofstatusimg = Mat::zeros(cv::Size(600, 50), CV_64F);
						putText(statusimg, textbuffer, Point(0, 80), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 3, 1);
						imshow("Status", statusimg);
						goto failure;
					}
				}

				if (doneflag == 1)
				{
					break;
				}

			}  // if ret success end
		} // inner while loop end
		
		pCam->EndAcquisition();
		pCam->DeInit();
		pCam = nullptr;
		
		// Clear camera list before releasing system
        camList.Clear();

        // Release system
        system->ReleaseInstance();
		

	} // end of try
	catch (Spinnaker::Exception &e)
	{
		cout << "Error: " << e.what() << endl;
		result = -1;
	}

#ifdef __unix__
	outfile << "% Parameters were - camgain, camtime, bpp, w , h , camspeed, usbtraffic, binvalue, bscanthreshold" << std::endl;
	outfile << "% " << camgain;
	outfile << ", " << camtime;
	outfile << ", " << bpp;
	outfile << ", " << w;
	outfile << ", " << h;
	outfile << ", " << camspeed;
	outfile << ", " << usbtraffic;
	outfile << ", " << binvalue;
	outfile << ", " << int(bscanthreshold);





#else
	  //imwrite("bscan.png", normfactorforsave*bscan);


	outfile << "camgain" << camgain;
	outfile << "camtime" << camtime;
	outfile << "bscanthreshold" << int(threshold);


#endif

	return 0;

failure:
	printf("Fatal error !! \n"); */
	return 1;
}

