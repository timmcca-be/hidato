#include "board.h"
#include "util.h"

HidatoBoard::HidatoBoard(int * board, int numRows, int numCols) {
    int numSlots = numRows * numCols;
    int * emptySlotMapping = new int[numSlots];
    int * prefilledNumbers = new int[numSlots];
    numEmptySlots = 0;
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
    availableNumbers = new int[numEmptySlots];
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
    comparisons = new int[numEmptySlots * 8]();
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

    int comparisonsSize = numEmptySlots * 8 * sizeof(int);
    cudaMalloc((void**) &comparisonsGpu, comparisonsSize);
    cudaMemcpy(comparisonsGpu, comparisons, comparisonsSize, cudaMemcpyHostToDevice);
}

HidatoBoard::~HidatoBoard() {
    delete[] comparisons;
    cudaFree(comparisonsGpu);
    delete[] availableNumbers;
}
