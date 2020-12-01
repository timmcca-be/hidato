#ifndef HIDATO_BOARD
#define HIDATO_BOARD

struct HidatoBoard {
    int * comparisons;
    int * comparisonsGpu;
    int * availableNumbers;
    int numEmptySlots;

    HidatoBoard(int * board, int numRows, int numCols);
    ~HidatoBoard();
};

#endif
