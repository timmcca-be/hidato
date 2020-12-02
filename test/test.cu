#include <iostream>
#include "../src/util.h"

bool testSort() {
    int values[] = {32, 36, 37, 31, 34, 38, 30, 25, 23, 12, 39, 29, 20, 28, 14, 19, 10, 15, 16, 8, 2, 17, 6, 3, 4};
    int valuesSorted[] = {2, 3, 4, 6, 8, 10, 12, 14, 15, 16, 17, 19, 20, 23, 25, 28, 29, 30, 31, 32, 34, 36, 37, 38, 39};
    sort(values, 25);
    for(int i = 0; i < 25; i++) {
        if(values[i] != valuesSorted[i]) {
            return false;
        }
    }
    return true;
}

int main(int argc, char ** argv) {
    std::cout << "Testing sort...\n";
    if(testSort()) {
        std::cout << "Pass!\n";
    } else {
        std::cout << "Fail :(\n";
    }
}
