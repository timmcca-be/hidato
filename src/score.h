#ifndef HIDATO_SCORE
#define HIDATO_SCORE

void scoreSolutions(int * comparisons, int * solutions, int numSolutionsToTest, int numEmptySlots, int * scores);
__global__ void scoreSolutionsParallel(int * comparisons, int * solutions, int * score);
__global__ void scoreSolutionsParallelByRow(int * comparisons, int * solutions, int * score);
__global__ void scoreSolutionsParallelBySolution(int * comparisons, int * solutions, int * score, int numEmptySlots);

#endif
