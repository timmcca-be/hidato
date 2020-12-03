#include "score.h"

void scoreSolutions(int * comparisons, int * solutions, int numSolutionsToTest, int numEmptySlots, int * scores) {
    for(int s = 0; s < numSolutionsToTest; s++) {
        int score = 0;
        int * solution = solutions + s * numEmptySlots;
        for(int i = 0; i < numEmptySlots; i++) {
            int oneAboveValue = solution[i] + 1;
            int oneBelowValue = oneAboveValue - 2;
            for(int j = 0; j < 8; j++) {
                int compareValue = comparisons[8 * i + j];
                if(compareValue == 0) {
                    break;
                }
                if(compareValue < 0) {
                    compareValue = solution[-compareValue - 1];
                }
                if(compareValue == oneAboveValue || compareValue == oneBelowValue) {
                    score += 1;
                }
            }
        }
        scores[s] = score;
    }
}

__global__ void
scoreSolutionsParallelBySolution(int * comparisons, int * solutions, int * scores, int numEmptySlots) {
    int score = 0;
    int * solution = solutions + blockIdx.x * numEmptySlots;
    for(int i = 0; i < numEmptySlots; i++) {
        int oneAboveValue = solution[i] + 1;
        int oneBelowValue = oneAboveValue - 2;
        for(int j = 0; j < 8; j++) {
            int compareValue = comparisons[8 * i + j];
            if(compareValue == 0) {
                break;
            }
            if(compareValue < 0) {
                compareValue = solution[-compareValue - 1];
            }
            if(compareValue == oneAboveValue || compareValue == oneBelowValue) {
                score += 1;
            }
        }
    }
    scores[blockIdx.x] = score;
}

__global__ void
scoreSolutionsParallelByRow(int * comparisons, int * solutions, int * scores) {
    extern __shared__ int comparisonResults[];

    int * solution = solutions + blockIdx.x * blockDim.x;
    int oneAboveValue = solution[threadIdx.x] + 1;
    int oneBelowValue = oneAboveValue - 2;
    int comparisonStart = threadIdx.x * 8;
    int comparisonEnd = comparisonStart + 8;
    int score = 0;
    for(int i = comparisonStart; i < comparisonEnd; i++) {
        int compareValue = comparisons[i];
        if(compareValue == 0) {
            break;
        }
        if(compareValue < 0) {
            compareValue = solution[-compareValue - 1];
        }
        if(compareValue == oneAboveValue || compareValue == oneBelowValue) {
            score++;
        }
    }
    comparisonResults[threadIdx.x] = score;
    __syncthreads();

    int lastI = blockDim.x;
    // parallel reduction - take the sum of all results in shared mem into the first element
    for(int i = blockDim.x >> 1; i > 0; i >>= 1) {
        if(threadIdx.x >= i) {
            return;
        }
        comparisonResults[threadIdx.x] += comparisonResults[threadIdx.x + i];
        // each iteration halves the remaining results - if the remaining chunk has an odd number of elements,
        //   the last thread needs to account for the last two elements
        if(threadIdx.x == i - 1) {
            int twoI = i + i;
            if(lastI > twoI) {
                comparisonResults[threadIdx.x] += comparisonResults[twoI];
            }
        }
        lastI = i;
        __syncthreads();
    }
    if(threadIdx.x == 0) {
        scores[blockIdx.x] = comparisonResults[0];
    }
}

__global__ void
scoreSolutionsParallel(int * comparisons, int * solutions, int * scores) {
    extern __shared__ int comparisonResults[];

    int threadIndex = threadIdx.y * blockDim.x + threadIdx.x;
    int numThreads = blockDim.x * blockDim.y;
    int compareValue = comparisons[threadIndex];
    if(compareValue == 0) {
        comparisonResults[threadIndex] = 0;
    } else {
        int * solution = solutions + blockDim.y * blockIdx.x;
        int value = solution[threadIdx.y];
        if(compareValue < 0) {
            compareValue = solution[-compareValue - 1];
        }
        comparisonResults[threadIndex] = value == compareValue + 1 || value == compareValue - 1;
    }
    __syncthreads();

    int lastI = numThreads;
    // parallel reduction - take the sum of all results in shared mem into the first element
    for(int i = numThreads >> 1; i > 0; i >>= 1) {
        if(threadIndex >= i) {
            return;
        }
        comparisonResults[threadIndex] += comparisonResults[threadIndex + i];
        // each iteration halves the remaining results - if the remaining chunk has an odd number of elements,
        //   the last thread needs to account for the last two elements
        if(threadIndex == i - 1) {
            int twoI = i + i;
            if(lastI > twoI) {
                comparisonResults[threadIndex] += comparisonResults[twoI];
            }
        }
        lastI = i;
        __syncthreads();
    }
    if(threadIndex == 0) {
        scores[blockIdx.x] = comparisonResults[0];
    }
}
