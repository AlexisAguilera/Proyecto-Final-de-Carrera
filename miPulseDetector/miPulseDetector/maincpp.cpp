#include "detectorDePulso.h"
//#include "ServerSocket.h"
#include <thread>  
#include <mutex>  
#include<winsock2.h>
#include <exception>
#include <iostream>
#include <sstream>
using namespace std;

//
// Main
//

detectorDePulso fd;


int main(int argc, const char** argv) {

	fd.run();

}