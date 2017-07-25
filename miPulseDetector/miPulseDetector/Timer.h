#pragma once

#define __cplusplus 201103L

#include <iostream>
#include <chrono>
#include <string.h>
using namespace std;
class Timer
{
public:
	typedef std::chrono::steady_clock Clock;
	typedef std::chrono::time_point<Clock> Time;

public:
	Timer()
		: t_start_(Clock::now())
	{
	}

	virtual ~Timer()
	{
	}

	inline void start()
	{
		t_start_ = Clock::now();
	}

	template <typename T>
	inline double getTimeDuration()
	{
		Time t = Clock::now();
		return std::chrono::duration_cast<T>(t - t_start_).count();
	}

	inline double getTimeNanoSec()
	{
		return getTimeDuration<std::chrono::nanoseconds>();
	}

	inline double getTimeMicroSec()
	{
		return getTimeDuration<std::chrono::microseconds>();
	}

	inline double getTimeMilliSec()
	{
		return getTimeDuration<std::chrono::milliseconds>();
	}

	inline double getTimeSec()
	{
		return getTimeDuration<std::chrono::seconds>();
	}


private:
	Time t_start_;

};
