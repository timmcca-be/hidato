#include <iostream>
#include <stdlib.h>

#include "timer.h"
#include "util.h"
#include "board.h"
#include "score.h"

int main(int argc, char ** argv) {
    srand(time(NULL));

    // 0 is a hole, -1 is an empty space
    int boardValues[] = {
        -1, 33, 35, -1, -1,  0,  0,  0,
        -1, -1, 24, 22, -1,  0,  0,  0,
        -1, -1, -1, 21, -1, -1,  0,  0,
        -1, 26, -1, 13, 40, 11,  0,  0,
        27, -1, -1, -1,  9, -1,  1,  0,
         0,  0, -1, -1, 18, -1, -1,  0,
         0,  0,  0,  0, -1,  7, -1, -1,
         0,  0,  0,  0,  0,  0,  5, -1
    };

    HidatoBoard board(boardValues, 8, 8);
    int * comparisons = board.comparisons;
    int numEmptySlots = board.numEmptySlots;

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
    int numSolutionsToTest = 10000;
    std::cout << "\n\nGenerating " << numSolutionsToTest - 1 << " random solutions to test, followed by the correct solution. The last score listed for each algorithm should be "
        << 2 * numEmptySlots << ".\n\n";
    int * solutions = new int[numEmptySlots * numSolutionsToTest];
    int solutionSize = numEmptySlots * sizeof(int);
    for(int i = 0; i < (numSolutionsToTest - 1) * numEmptySlots; i += numEmptySlots) {
        int * solution = solutions + i;
        memcpy(solution, board.availableNumbers, solutionSize);
        shuffle(solution, numEmptySlots);
    }

    memcpy(solutions + (numSolutionsToTest - 1) * numEmptySlots, correctSolution, solutionSize);

    int * scores = new int[numSolutionsToTest];

    CpuTimer cpuTimer;
    cpuTimer.start();
    scoreSolutions(comparisons, solutions, numSolutionsToTest, numEmptySlots, scores);
    cpuTimer.stop();
    std::cout << "Scores by sequential algorithm:\n";
    for(int i = 0; i < numSolutionsToTest; i++) {
        std::cout << scores[i] << " ";
    }
    std::cout << "\n\nSequential algorithm: " << cpuTimer.elapsed() << " ms\n";

    int * solutionsDevice;
    int solutionsSize = numEmptySlots * numSolutionsToTest * sizeof(int);
    cudaMalloc((void**) &solutionsDevice, solutionsSize);
    cudaMemcpy(solutionsDevice, solutions, solutionsSize, cudaMemcpyHostToDevice);
    delete[] solutions;

    int * scoresDevice;
    int scoresSize = numSolutionsToTest * sizeof(int);
    cudaMalloc((void**) &scoresDevice, scoresSize);

    dim3 grid(8, numEmptySlots);
    dim3 blockGrid(numSolutionsToTest);
    int sharedMemorySize = 8 * numEmptySlots * sizeof(int);
    GpuTimer gpuTimer;
    gpuTimer.start();
    scoreSolutionsParallel<<<blockGrid, grid, sharedMemorySize>>>(board.comparisonsGpu, solutionsDevice, scoresDevice);
    gpuTimer.stop();

    int * parallelScores = new int[numSolutionsToTest];
    cudaMemcpy(parallelScores, scoresDevice, scoresSize, cudaMemcpyDeviceToHost);
    bool resultsCorrect = true;
    for(int i = 0; i < numSolutionsToTest; i++) {
        if(parallelScores[i] != scores[i]) {
            resultsCorrect = false;
            break;
        }
    }
    if(resultsCorrect) {
        std::cout << "Parallel by cell algorithm: " << gpuTimer.elapsed() << " ms\n";
    } else {
        std::cout << "Parallel by cell algorithm returned incorrect answer\n";
    }

    dim3 byRowGrid(numEmptySlots);
    sharedMemorySize = numEmptySlots * sizeof(int);
    gpuTimer.start();
    scoreSolutionsParallelByRow<<<blockGrid, byRowGrid, sharedMemorySize>>>(board.comparisonsGpu, solutionsDevice, scoresDevice);
    gpuTimer.stop();

    cudaMemcpy(parallelScores, scoresDevice, scoresSize, cudaMemcpyDeviceToHost);
    resultsCorrect = true;
    for(int i = 0; i < numSolutionsToTest; i++) {
        if(parallelScores[i] != scores[i]) {
            resultsCorrect = false;
            break;
        }
    }
    if(resultsCorrect) {
        std::cout << "Parallel by row algorithm: " << gpuTimer.elapsed() << " ms\n";
    } else {
        std::cout << "Parallel by row algorithm returned incorrect answer\n";
    }

    gpuTimer.start();
    scoreSolutionsParallelBySolution<<<blockGrid, 1, sharedMemorySize>>>(board.comparisonsGpu, solutionsDevice, scoresDevice, numEmptySlots);
    gpuTimer.stop();

    cudaMemcpy(parallelScores, scoresDevice, scoresSize, cudaMemcpyDeviceToHost);
    resultsCorrect = true;
    for(int i = 0; i < numSolutionsToTest; i++) {
        if(parallelScores[i] != scores[i]) {
            resultsCorrect = false;
            break;
        }
    }
    if(resultsCorrect) {
        std::cout << "Parallel by solution algorithm: " << gpuTimer.elapsed() << " ms\n";
    } else {
        std::cout << "Parallel by solution algorithm returned incorrect answer\n";
    }

    std::cout << "\n";

    cudaFree(solutionsDevice);
    cudaFree(scoresDevice);
    delete[] scores;
    delete[] parallelScores;
}
