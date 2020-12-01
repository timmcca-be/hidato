#include "score.h"

void scoreSolutions(int * comparisons, int * solutions, int numSolutionsToTest, int numEmptySlots, int * scores) {
    for(int s = 0; s < numSolutionsToTest; s++) {
        int score = 0;
        int * solution = solutions + s * numEmptySlots;
        for(int i = 0; i < numEmptySlots; i++) {
            int value = solution[i];
            for(int j = 0; j < 8; j++) {
                int compareValue = comparisons[8 * i + j];
                if(compareValue == 0) {
                    continue;
                } else if(compareValue < 0) {
                    compareValue = solution[-compareValue - 1];
                }
                if(value == compareValue + 1 || value == compareValue - 1) {
                    score += 1;
                }
            }
        }
        scores[s] = score;
    }
}

__global__ void
scoreSolutionsParallel(int * comparisons, int * solutions, int * score) {
    extern __shared__ int comparisonResults[];

    int threadIndex = threadIdx.y * blockDim.x + threadIdx.x;
    int numThreads = blockDim.x * blockDim.y;
    int compareValue = comparisons[threadIndex];
    if(compareValue == 0) {
        comparisonResults[threadIndex] = 0;
    } else {
        int * solution = solutions + numThreads * blockIdx.x;
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
        score[blockIdx.x] = comparisonResults[0];
    }
}
