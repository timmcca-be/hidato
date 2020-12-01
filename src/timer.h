#include <chrono>

#ifndef __HIDATO_TIMER_H__
#define __HIDATO_TIMER_H__

class GpuTimer {
private:
      cudaEvent_t startEvent;
      cudaEvent_t stopEvent;
public:
      GpuTimer();
      ~GpuTimer();
      void start();
      void stop();
      float elapsed();
};

class CpuTimer {
private:
      std::chrono::high_resolution_clock::time_point startTime;
      std::chrono::high_resolution_clock::time_point stopTime;
public:
      void start();
      void stop();
      float elapsed();
};

#endif
