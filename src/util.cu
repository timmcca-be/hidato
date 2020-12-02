#include <utility>

// quicksort
void sort(int * values, int length) {
    if(length <= 1) {
        return;
    }
    int pivot = values[0];
    int center = 1;
    for(int i = 1; i < length; i++) {
        if(values[i] < pivot) {
            std::swap(values[i], values[center++]);
        }
    }
    values[0] = values[center - 1];
    values[center - 1] = pivot;
    sort(values, center - 1);
    sort(values + center, length - center);
}

// fisher-yates shuffle
void shuffle(int * values, int length) {
    for(int i = length - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        std::swap(values[i], values[j]);
    }
}
