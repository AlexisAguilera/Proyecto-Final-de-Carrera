#define _USE_MATH_DEFINES
#include <cmath> 
#include<winsock2.h>
#include <exception>
#include <iostream>
#include <sstream>
#include "detectorDePulso.h"
#pragma comment(lib,"ws2_32.lib") //Winsock Library

void dump(const string& label, vector<double> data) {
	printf("%s", label.c_str());
	for (int i = 0; i < data.size(); ++i) {
		printf("[%lf]", data[i]);
	}
	printf("\n");
}
cv::VideoCapture camera;

void detectorDePulso::run() {

	// detector de rostros
	cv::CascadeClassifier faceClassifier;
	if (!faceClassifier.load("C:\\OpenCV-3.2.0\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_alt.xml")) {
		cerr << "Unable to load face classifier.\n";
		exit(1);
	}

	
	if (!camera.open(0)) {
		cerr << "Unable to initialise camera.\n";
		exit(1);
	}
	
	const char* windowName = "BPM Monitor";
	cv::namedWindow(windowName, 1);
	cv::moveWindow(windowName, 0, 0);
		//socket
	
		WSADATA wsa;
		SOCKET s, new_socket;
		char buf[256];
		struct sockaddr_in server, client;
		int c;

		printf("\nInitialising Winsock...");
		if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0)
		{
			printf("Failed. Error Code : %d", WSAGetLastError());
		}

		printf("Initialised.\n");
		
		//Create a socket
		if ((s = socket(AF_INET, SOCK_STREAM, 0)) == INVALID_SOCKET)
		{
			printf("Could not create socket : %d", WSAGetLastError());
		}

		printf("Socket created.\n");

		//Prepare the sockaddr_in structure
		server.sin_family = AF_INET;
		server.sin_addr.s_addr = INADDR_ANY;
		server.sin_port = htons(8888);

		//Bind


		if (::bind(s, (struct sockaddr *)&server, sizeof(server)) == SOCKET_ERROR)
		{
			printf("Bind failed with error code : %d", WSAGetLastError());
		}

		puts("Bind done");


		//Listen to incoming connections
		listen(s, 3);

		//Accept and incoming connection
		puts("Waiting for incoming connections...");


		c = sizeof(struct sockaddr_in);
		new_socket = accept(s, (struct sockaddr *)&client, &c);
		if (new_socket == INVALID_SOCKET)
		{
			printf("accept failed with error code : %d", WSAGetLastError());
		}

		puts("Connection accepted");

		closesocket(s);
		//WSACleanup();
		//memset(buf, '/0', 1024);

		int len, h, w, l;
		if (len = recv(new_socket, buf, sizeof(buf), 0) != -1) {
			char *length, *width, *high;
			length = strtok(buf, " ");
			width = strtok(NULL, " ");
			high = strtok(NULL, " ");
			printf("\nlength = %s \nwidth = %s \nhigh = %s", length, width, high);
			l = atoi(length);
			h = atoi(high);
			w = atoi(width);
			//char* msg = (char*) malloc(sizeof(char) *4);
			char msg[] = { 'o', 'k', '\0' };
			send(new_socket, msg, sizeof(msg), 0);

			char* c = (char*)malloc(sizeof(char) * l);
			//std::vector <char> v;

		// ventana para mostrar el video
	
	_start = boost::chrono::system_clock::now();
	// se inicializa el reloj


		// Video processing loop
	bool processing = true;
	bool monitoring = false;

	while (processing) {
		cv::Mat frameGreyscale, daGrayFace, scaleddaGrayFace;
	
		
		cv::Mat  img = cv::Mat::zeros(h + h / 2, w, CV_8UC1);
		int  imgSize = img.total()*img.elemSize();
		uchar* sockData = (uchar*)malloc(sizeof(uchar)*imgSize);
		//Receive data here
		int bytes = 0;
		for (int i = 0; i < l; i += bytes) {
			if ((bytes = recv(new_socket, (char*)sockData + i, imgSize - i, 0)) == -1) {
				printf("recv failed");
				break;
			}
		}

		cv::Mat frameOriginal(h + h / 2, w, CV_8UC1, sockData);
		cv::cvtColor(frameOriginal, frameOriginal, cv::COLOR_YUV2RGB_NV12);
	
			//revisar esta espera
			//	cv::waitKey(30);


			if (!frameOriginal.empty()) {
				cv::Scalar color(255, 255, 0, 0);
				//cv::resize(frameOriginal, frameOriginal, cv::Size(800, 600), CV_INTER_NN);
				//se convierte imagen a escala de grises
				cv::flip(frameOriginal, frameOriginal, 1);
				cv::cvtColor(frameOriginal, frameGreyscale, CV_BGR2GRAY);
				cv::equalizeHist(frameGreyscale, frameGreyscale);
				// Detector de rostros
				vector<cv::Rect> faces;
				faceClassifier.detectMultiScale(frameGreyscale, faces, 1.1, 3,
					CV_HAAR_FIND_BIGGEST_OBJECT | CV_HAAR_SCALE_IMAGE,
					cv::Size(30, 30));

				// se detectaron rostros
				// se dibujan rectángulo en los rostros detectados y sus frentes 
				if (!faces.empty()) {
					cv::Rect fh;
					cv::Rect grabbedFace;
					//for (int i = 0; i < faces.size(); i++) {
						//dibuja un rectángulo verde en el rostro sobre la imagen original
					if (!monitoring) {
						getFrente(faces[0], fh);
						
						getFrente(faces[0], _frente); // se guarda la frente del primer rostro capturado
						cv::rectangle(frameOriginal, faces[0], cv::Scalar(255, 0, 0, 0), 1, 8, 0);
						cv::putText(frameOriginal, "Rostro", cv::Point(faces[0].x, faces[0].y), CV_FONT_HERSHEY_PLAIN, 1.2, color);
						cv::rectangle(frameOriginal, fh, cv::Scalar(0, 255, 0, 0), 1, 8, 0);
						cv::putText(frameOriginal, "Frente", cv::Point(fh.x, fh.y), CV_FONT_HERSHEY_PLAIN, 1.2, color);
						cv::putText(frameOriginal, "Presione 'R' para comenzar a monitorear", cv::Point(10, 40), CV_FONT_HERSHEY_PLAIN, 1.2, color);
					}
					else { //empieza el monitoreo
						// Get image data for the forehead for BPM monitoring
						cv::Mat fhimg = frameOriginal(_frente);
						cv::rectangle(frameOriginal, _frente, cv::Scalar(0, 255, 255, 0), 1, 8, 0);
					
						//cv::putText(frameOriginal, "BMP Area", cv::Point(_frente.x, _frente.y), CV_FONT_HERSHEY_PLAIN, 1.2, color);
						cv::putText(frameOriginal, "Presione 'S' para terminar de monitorear", cv::Point(10, 40), CV_FONT_HERSHEY_PLAIN, 1.2, color);
						double espera =_means.size();
						float aux = (espera / MAX_SAMPLES)*100;
						espera = trunc(aux);
						if (espera < 100) {
							char buffer[50];
							int n = snprintf(buffer, 50, "cargando %0.0lf %c ... ", espera,'%');
							cv::putText(frameOriginal, buffer, cv::Point(_frente.x + 100, _frente.y + 25), CV_FONT_HERSHEY_PLAIN, 1.2, color);
						}
			
						pdata = estimateBPM(fhimg);
						char bpmbuffer[50];
						int n = sprintf(bpmbuffer, "estimado: %0.1lf bpm", pdata);
						cv::putText(frameOriginal, bpmbuffer, cv::Point(_frente.x, _frente.y), CV_FONT_HERSHEY_PLAIN, 1.2, color);

					}

				}
				// Show video image + annotations on window

				cv::putText(frameOriginal, "Presione 'Esc' para salir", cv::Point(10, 20), CV_FONT_HERSHEY_PLAIN, 1.2, color);
			
				cv::imshow(windowName, frameOriginal);

				// Pause for 10m and read key, if any
				switch (tolower(cv::waitKey(1))) {
				case KEY_ESCAPE:
					processing = false;
					WSACleanup();
					break;
				case KEY_R:
					if (_frente.area() != 0) {
						monitoring = true;
					}
					break;
				case KEY_S:
					monitoring = false;
					clearBuffers();
					break;
				}
			}
		}
	}
		return;

}

void detectorDePulso::getFrente(const cv::Rect& face, cv::Rect& forehead) {
		getSubface(face, forehead, 0.50, 0.18, 0.25, 0.15);
		//getSubface(face, forehead, 0.50, 0.13, (0.45), 0.20);
		return;
	}

void detectorDePulso::getSubface(const cv::Rect& face, cv::Rect& sub, float sf_x, float sf_y, float sf_w, float sf_h) {
		//assert(face.height != 0 && face.width != 0);
		//assert(sf_w > 0.0 && sf_y > 0.0 && sf_w > 0.0 && sf_h > 0.0);
		sub.x = face.x + face.width * sf_x - (face.width * sf_w / 2.0);
		sub.y = face.y + face.height * sf_y - (face.height * sf_h / 2.0);
		sub.width = face.width * sf_w;
		sub.height = face.height * sf_h;
		return;
	}


double detectorDePulso::calculate_mean(const cv::Mat& image) {
	cv::Scalar means = cv::mean(image);
			//BLUE MEAN     GREEN MEAN    RED MEAN	
	//return (means.val[0] + means.val[1] + means.val[2]) / 3;
	return means.val[1]; //solo canal verde
}


double detectorDePulso::timestamp() {
	boost::chrono::duration<double> seconds = boost::chrono::system_clock::now() - _start;
	return seconds.count();
}

vector<double> detectorDePulso::hammingWindow(int M) {
	vector<double> window(M);
	if (M == 1) {
		window[0] = 1.0;
	}
	else {
		for (int n = 0; n < M; ++n) {
			window[n] = 0.54 - 0.46 * cos((2 * M_PI * n) / (M - 1));
		}
	}
	return window;
}


vector<double> detectorDePulso::interp(vector<double> interp_x, vector<double> data_x, vector<double> data_y) {
	assert(data_x.size() == data_y.size());
	vector<double> interp_y(interp_x.size());
	vector<double> interpRes;

	// GSL function expects an array
	double* data_y_array = (double*)malloc(sizeof(double)*data_y.size());
	double* data_x_array = (double*)malloc(sizeof(double)*data_x.size());
	copy(data_y.begin(), data_y.end(), data_y_array);
	copy(data_x.begin(), data_x.end(), data_x_array);

	double yi;
	int L = interp_x.size();

	gsl_interp_accel *acc = gsl_interp_accel_alloc();
	gsl_spline *spline = gsl_spline_alloc(gsl_interp_cspline, L);
	gsl_spline_init(spline, data_x_array, data_y_array, L);

	// BUGFIX: Need to iterate throuh given x-interpolation range
	for (int xi = 0; xi < interp_x.size(); xi++)
	{
		yi = gsl_spline_eval(spline, interp_x[xi], acc);
		interpRes.push_back(yi);
		//printf ("%g\n", yi);
	}

	gsl_spline_free(spline);
	gsl_interp_accel_free(acc);

	return interpRes;
}

vector<gsl_complex> detectorDePulso::fft_transform(vector<double>& samples) {
	int size = samples.size();
	double* data = (double*)malloc(sizeof(double)*size);
	copy(samples.begin(), samples.end(), data);
	// Transform to fft
	//gsl_fft_real_workspace* work = gsl_fft_real_workspace_alloc(size);
	//gsl_fft_real_wavetable* real = gsl_fft_real_wavetable_alloc(size);
	//gsl_fft_real_transform(data, 1, size, real, work);
	//gsl_fft_real_wavetable_free(real);
	//gsl_fft_real_workspace_free(work);
	gsl_fft_real_radix2_transform(data, 1, size);

	//una loquita decía que calcule la inversa nuevamente para que se vaya el ruido
	//printf("\nel tamaño de data es %d", size);
	//gsl_fft_complex_radix2_inverse(data, 1, size);


	// Unpack complex numbers
	gsl_complex* unpacked = (gsl_complex*)malloc(sizeof(gsl_complex)* size);
	gsl_fft_halfcomplex_unpack(data, (double *)unpacked, 1, size);
	// Copy to  a vector
	int unpacked_size = size / 2 + 1;
	vector<gsl_complex> output(unpacked, unpacked + unpacked_size);
	return output;
}



vector<double> detectorDePulso::calculate_complex_abs(vector<gsl_complex> cvalues) {
	// Calculate absolute value of a given complex number
	vector<double> output(cvalues.size());
	for (int i = 0; i < cvalues.size(); i++) {
		output[i] = gsl_complex_abs(cvalues[i]);
	}
	return output;
}

void detectorDePulso::list_multiply(vector<double>& data, double value) {
	for (int i = 0; i < data.size(); ++i) {
		data[i] *= value;
	}
}

vector<double> detectorDePulso::arange(int stop) {
	vector<double> range(stop);
	for (int i = 0; i < stop; i++) {
		range[i] = i;
	}
	return range;
}


//
// función de procesamiento prinicipal
//
double detectorDePulso::estimateBPM(const cv::Mat& skin) {
	//calcula la media de R, G y B.
	_means.push_back(calculate_mean(skin));
	_times.push_back(timestamp());


	double pdata;
	int sampleSize = _means.size();
	//printf("\n %d/64 ", sampleSize);
	// Check Point
	assert(_times.size() == sampleSize);

	// Si no hay suficientes muestras no se mide nada
	if (sampleSize <= MIN_SAMPLES) {
		pdata = 0;
		return pdata;
	}
	// si hay demasiadas muestras, se descartan las más viejas
	if (sampleSize > MAX_SAMPLES) {
		list_trimfront(_means, MAX_SAMPLES);
		list_trimfront(_times, MAX_SAMPLES);
		list_trimfront(_bpms, MAX_SAMPLES);
		sampleSize = MAX_SAMPLES;
	}
	// FPS

	_fps = sampleSize / (_times.back() - _times.front());
	vector<double> even_times = linspace(_times.front(), _times.back(), sampleSize);
	vector<double> interpolated = interp(even_times, _times, _means);

//	dump("interpolated" , interpolated);

	vector<double> hamming = hammingWindow(sampleSize);

	list_multiply_vector(interpolated, hamming);

	double totalMean = list_mean(interpolated);

	//cout << "total Mean = " << totalMean << endl;
   //normalizando señal
	double aux = list_subtract(interpolated, totalMean); //retorna  la suma del cuadrado de la media menos la medida
	double desvio = sqrt(aux / (interpolated.size())); //desvío estandar
	list_normalizar(interpolated, totalMean, desvio);

	//acá falta un paso, descompone las señales en los 3 canales y usa el algoritmo de Cardoso para el ICA


	// transformada de fourier
	vector<gsl_complex> fftraw = fft_transform(interpolated);

	//obtiene los valores absolutos de los coeficientes de la FFT.
	_fftabs = calculate_complex_abs(fftraw);
	//frecuencias con valores entre 0 y L/2+1
	_frecuencias = arange((sampleSize / 2) + 1);
	
	list_multiply(_frecuencias, (_fps*75)/ sampleSize); //es una pequeña corrección, en teoría se multiplica por 60

	//vector<double> freqs(_frecuencias);
	
	//filtra las frecuencias de salida, se queda con aquellas que coincide con los latidos entre 50 y 180 (0.4,4hz)
	vector<double> fitered_indices = list_filter(_frecuencias, BPM_FILTER_LOW, BPM_FILTER_HIGH);
	_fftabs = list_pruned(_fftabs, fitered_indices);
	_frecuencias = list_pruned(_frecuencias, fitered_indices);
	
	if (_fftabs.size() > 0) {
		int max = list_argmax(_fftabs);
		_bpm = _frecuencias[max];
		cout << "esto da bpm = " << _bpm << endl;
		_bpms.push_back(_bpm);
	}
	else {
		_bpm = 0;
	}
	pdata = _bpm;

	return pdata;

}



//operaciones sobre vectores

void detectorDePulso::list_multiply_vector(vector<double>& data, vector<double>& mult) {
	assert(data.size() == mult.size());
	for (int i = 0; i < data.size(); ++i) {
		data[i] *= mult[i];
	}
}

void detectorDePulso::list_normalizar(vector<double>& data, double media, double desvio) {
	for (int i = 0; i < data.size(); ++i) {
		data[i] = (data[i] - media) / desvio;
	}
}

double detectorDePulso::list_subtract(vector<double>& data, double value) {
	double toRet = 0;
	double aux;
	for (int i = 0; i < data.size(); ++i) {
		aux = data[i] - value;
		toRet += aux * data[i];
	}
	return toRet;
}


double detectorDePulso::list_mean(vector<double>& data) {
	assert(!data.empty());
	boost::accumulators::accumulator_set<double, boost::accumulators::stats<boost::accumulators::tag::mean> > acc;
	for_each(data.begin(), data.end(), boost::bind<void>(boost::ref(acc), _1));
	return boost::accumulators::mean(acc);
}

vector<double> detectorDePulso::list_filter(vector<double>& data, double low, double high) {
	vector<double> indices;
	for (int i = 0; i < data.size(); ++i) {
		if (data[i] >= low && data[i] <= high) {
			indices.push_back(i);
		}
	}
	return indices;
}

void detectorDePulso::clearBuffers()
{

	_means.clear();
	_times.clear();
	_fftabs.clear();
	_frecuencias.clear();
	_pruned.clear();
	_prunedfreq.clear();
	_bpms.clear();
	gap = 0;

}

vector<double> detectorDePulso::list_pruned(vector<double>& data, vector<double>& indices) {
	vector<double> pruned;
	for (int i = 0; i < indices.size(); ++i) {
		assert(indices[i] >= 0 && indices[i] < data.size());
		pruned.push_back(data[indices[i]]);
	}
	return pruned;
}

void detectorDePulso::list_trimfront(vector<double>& list, int limit) {
	int exceso = list.size() - limit;
	if (exceso > 0) {
		list.erase(list.begin(), list.begin() + exceso);
	}
}


vector<double> detectorDePulso::linspace(double start, double end, int count) {
	vector<double> intervals(count);
	double gap = (end - start) / (count - 1);
	intervals[0] = start;
	for (int i = 1; i < (count - 1); ++i) {
		intervals[i] = intervals[i - 1] + gap;
	}
	intervals[count - 1] = end;
	return intervals;
}


int detectorDePulso::list_argmax(vector<double>& data) {
	int indmax = 0;
	double argmax = 0;

	for (int i = 0; i < data.size(); ++i) {

		if (data[i] > argmax) {

			argmax = data[i];
			indmax = i;
		}
	}
	return indmax;
}
