#include "Timer.h"
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/core/core.hpp"
#include <stdio.h>
#include <cstdio>
#include <algorithm>
#include <iterator>
#include <list> 
#include <fstream>
#include <string.h>
#include <stdlib.h>

#include <boost/bind.hpp>
#include <boost/ref.hpp>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/mean.hpp>



#include <gsl/gsl_errno.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_complex.h>
#include <gsl/gsl_complex_math.h>
#include <gsl/gsl_fft_real.h>
#include <gsl/gsl_fft_complex.h>
#include <gsl/gsl_fft_halfcomplex.h>
#include <boost/chrono.hpp>

#include <iostream>
#include <string>
#include <vector>

#include <complex>

#define KEY_ESCAPE	((char) 27)
#define KEY_S		('s')
#define KEY_R		('r')
#define PI 3.14159265

#define MAX_SAMPLES		(16)
#define MIN_SAMPLES 	(16)
#define BPM_FILTER_LOW	(50)
#define BPM_FILTER_HIGH	(180)





using namespace std;

class detectorDePulso {


private:
	cv::Mat _face;
	cv::Rect _frente;
	vector<double> _means;
	vector<double> _times;
	vector<double> _fftabs;
	vector<double> _frecuencias;
	vector<double> _pruned;
	vector<double> _prunedfreq;
	vector<double> _bpms;
	double _fps;
	double _bpm;
	double gap;
	const static int idx = 1;
	boost::chrono::system_clock::time_point _start;
	double pdata;

public:

	void run();

private:
	void getFrente(const cv::Rect& face, cv::Rect& frente);
	void getSubface(const cv::Rect& face, cv::Rect& sub, float fh_x, float fh_y, float fh_w, float fh_h);
	double estimateBPM(const cv::Mat& fhimg);
	//double getSubface_means(const cv::Mat& image, cv::Rect& frente);
	double calculate_mean(const cv::Mat& image);

	double timestamp();
	void clearBuffers();

	vector<double> hammingWindow(int M);
	vector<double> interp(vector<double> interp_x, vector<double> data_x, vector<double> data_y);

	vector<gsl_complex> fft_transform(vector<double>& samples);

//	vector<double> calculate_complex_angle(vector<gsl_complex> cvalues);
	vector<double> calculate_complex_abs(vector<gsl_complex> cvalues);

	// Operaciones sobre listas
	double list_mean(vector<double>& data);
	void list_multiply(vector<double>& data, double value);
	void list_multiply_vector(vector<double>& data, vector<double>& mult);
	double list_subtract(vector<double>& data, double value);
	void list_trimfront(vector<double>& list, int limit);
	vector<double> list_pruned(vector<double>& data, vector<double>& index);
	vector<double> list_filter(vector<double>& data, double low, double high);
	int list_argmax(vector<double>& data);
	void list_normalizar(vector<double>& data, double media, double desvio);

	vector<double> linspace(double start, double end, int count);
	vector<double> arange(int stop);




};