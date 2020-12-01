#include <iostream>
#include <time.h>
#include <stdlib.h>
#include "timer.h"
#include "util.h"

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

int main(int argc, char ** argv) {
    srand(time(NULL));

    int numRows = 8;
    int numCols = 8;
    // 0 is a hole, -1 is an empty space
    int board[] = {
        -1, 33, 35, -1, -1,  0,  0,  0,
        -1, -1, 24, 22, -1,  0,  0,  0,
        -1, -1, -1, 21, -1, -1,  0,  0,
        -1, 26, -1, 13, 40, 11,  0,  0,
        27, -1, -1, -1,  9, -1,  1,  0,
         0,  0, -1, -1, 18, -1, -1,  0,
         0,  0,  0,  0, -1,  7, -1, -1,
         0,  0,  0,  0,  0,  0,  5, -1
    };

    int numSlots = numRows * numCols;
    int * emptySlotMapping = new int[numSlots];
    int * prefilledNumbers = new int[numSlots];
    int numEmptySlots = 0;
    int numPrefilledSlots = 0;
    for(int i = 0; i < numRows * numCols; i++) {
        if(board[i] == -1) {
            emptySlotMapping[numEmptySlots++] = i;
            // edit the board so that the first empty slot is -1, the second is -2, etc.
            board[i] = -numEmptySlots;
        } else if(board[i] > 0) {
            prefilledNumbers[numPrefilledSlots++] = board[i];
        }
    }

    sort(prefilledNumbers, numPrefilledSlots);
    int prefilledIndex = 0;
    int availableIndex = 0;
    int * availableNumbers = new int[numEmptySlots];
    for(int i = 1; i <= numEmptySlots + numPrefilledSlots; i++) {
        if(i == prefilledNumbers[prefilledIndex]) {
            prefilledIndex += 1;
        } else {
            availableNumbers[availableIndex] = i;
            availableIndex += 1;
        }
    }

    delete[] prefilledNumbers;

    // think of this as a numEmptySlots x 8 matrix
    // each row contains the elements that need to be compared to the associated slot in the solution
    // positive values are literal, negative values reference elements in the solution, 0s are ignored
    int * comparisons = new int[numEmptySlots * 8]();
    for(int i = 0; i < numEmptySlots; i++) {
        int index = emptySlotMapping[i];
        bool canGoLeft = index % numCols > 0;
        bool canGoRight = (index + 1) % numCols > 0;
        bool canGoUp = index >= numCols;
        bool canGoDown = (index / numCols) + 1 < numRows;
        int comparisonIndex = 8 * i;
        if(canGoUp) {
            comparisons[comparisonIndex++] = board[index - numCols];
            if(canGoLeft) {
                comparisons[comparisonIndex++] = board[index - numCols - 1];
            }
            if(canGoRight) {
                comparisons[comparisonIndex++] = board[index - numCols + 1];
            }
        }
        if(canGoLeft) {
            comparisons[comparisonIndex++] = board[index - 1];
        }
        if(canGoRight) {
            comparisons[comparisonIndex++] = board[index + 1];
        }
        if(canGoDown) {
            comparisons[comparisonIndex++] = board[index + numCols];
            if(canGoLeft) {
                comparisons[comparisonIndex++] = board[index + numCols - 1];
            }
            if(canGoRight) {
                comparisons[comparisonIndex] = board[index + numCols + 1];
            }
        }
    }

    delete[] emptySlotMapping;

    std::cout << "Comparisons used to determine score (L# refers to the literal number #, S# refers to the value in slot #):\n";
    for(int i = 0; i < numEmptySlots; i++) {
        std::cout << "Slot " << i + 1 << ": \t";
        for(int j = 0; j < 8; j++) {
            int value = comparisons[8 * i + j];
            if(value > 0) {
                std::cout << "L" << value << "\t";
            } else if(value < 0) {
                std::cout << "S" << -value << "\t";
            }
        }
        std::cout << "\n";
    }

    int correctSolution[] = {32, 36, 37, 31, 34, 38, 30, 25, 23, 12, 39, 29, 20, 28, 14, 19, 10, 15, 16, 8, 2, 17, 6, 3, 4};
    std::cout << "\nKnown correct solution: ";
    for(int i = 0; i < numEmptySlots; i++) {
        std::cout << correctSolution[i] << " ";
    }

    // Generate a bunch of random solutions
    int numSolutionsToTest = 2000;
    std::cout << "\n\nGenerating " << numSolutionsToTest - 1 << " random solutions to test, followed by the correct solution. The last score listed for each algorithm should be "
        << 2 * numEmptySlots << ".\n\n";
    int * solutions = new int[numEmptySlots * numSolutionsToTest];
    int solutionSize = numEmptySlots * sizeof(int);
    for(int i = 0; i < (numSolutionsToTest - 1) * numEmptySlots; i += numEmptySlots) {
        int * solution = solutions + i;
        memcpy(solution, availableNumbers, solutionSize);
        shuffle(solution, numEmptySlots);
    }
    delete[] availableNumbers;
    memcpy(solutions + (numSolutionsToTest - 1) * numEmptySlots, correctSolution, solutionSize);

    int * scores = new int[numSolutionsToTest];

    CpuTimer cpuTimer;
    cpuTimer.start();
    scoreSolutions(comparisons, solutions, numSolutionsToTest, numEmptySlots, scores);
    cpuTimer.stop();
    std::cout << "Scores by sequential algorithm (" << cpuTimer.elapsed() << " ms):\n";
    for(int i = 0; i < numSolutionsToTest; i++) {
        std::cout << scores[i] << " ";
    }

    int * comparisonsDevice;
    int numComparisons = numEmptySlots * 8;
    int comparisonsSize = numComparisons * sizeof(int);
    cudaMalloc((void**) &comparisonsDevice, comparisonsSize);
    cudaMemcpy(comparisonsDevice, comparisons, comparisonsSize, cudaMemcpyHostToDevice);
    delete[] comparisons;

    int * solutionsDevice;
    int solutionsSize = numEmptySlots * numSolutionsToTest * sizeof(int);
    cudaMalloc((void**) &solutionsDevice, solutionsSize);
    cudaMemcpy(solutionsDevice, solutions, solutionsSize, cudaMemcpyHostToDevice);
    delete[] solutions;

    int * scoresDevice;
    cudaMalloc((void**) &scoresDevice, numSolutionsToTest * sizeof(int));

    dim3 grid(8, numEmptySlots);
    dim3 blockGrid(numSolutionsToTest);
    int sharedMemorySize = numComparisons * sizeof(int);
    GpuTimer gpuTimer;
    gpuTimer.start();
    scoreSolutionsParallel<<<blockGrid, grid, sharedMemorySize>>>(comparisonsDevice, solutionsDevice, scoresDevice);
    gpuTimer.stop();

    cudaFree(solutionsDevice);
    cudaFree(comparisonsDevice);

    cudaMemcpy(scores, scoresDevice, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(scoresDevice);

    std::cout << "\n\nScores by parallel algorithm (" << gpuTimer.elapsed() << " ms):\n";
    for(int i = 0; i < numSolutionsToTest; i++) {
        std::cout << scores[i] << " ";
    }
    delete[] scores;
    std::cout << "\n\n";
}
