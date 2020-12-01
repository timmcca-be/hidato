#include "timer.h"

#include <chrono>
#include "util.h"


GpuTimer::GpuTimer() {
      cudaEventCreate(&startEvent);
      cudaEventCreate(&stopEvent);
}

GpuTimer::~GpuTimer() {
      cudaEventDestroy(startEvent);
      cudaEventDestroy(stopEvent);
}

void GpuTimer::start() {
      cudaEventRecord(startEvent, 0);
}

void GpuTimer::stop() {
      cudaEventRecord(stopEvent, 0);
}

float GpuTimer::elapsed() {
      float elapsedTime;
      cudaEventSynchronize(stopEvent);
      cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
      return elapsedTime;
}

void CpuTimer::start() {
      startTime = std::chrono::high_resolution_clock::now();
}

void CpuTimer::stop() {
      stopTime = std::chrono::high_resolution_clock::now();
}

float CpuTimer::elapsed() {
      return std::chrono::duration_cast<std::chrono::nanoseconds>(stopTime - startTime).count() / 1000000.0f;
}
