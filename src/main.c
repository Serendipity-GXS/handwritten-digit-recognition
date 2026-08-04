// main.c — temporary test entry for the data layer.
#include <stdio.h>
#include "data.h"

int main(void)
{
    MnistSet test;

    int err = mnist_load(&test,
                         "data\\t10k-images-idx3-ubyte",
                         "data\\t10k-labels-idx1-ubyte");
    if (err != MNIST_OK) {
        printf("mnist_load failed, err = %d\n", err);
        return 1;
    }

    printf("n = %d, %d x %d\n", test.n, test.height, test.width);

    for (int i = 0; i < 3; i++) {
        printf("\n--- sample %d, label = %d ---\n", i, test.labels[i]);
        print_ascii_image(test.images + i * test.height * test.width,
                          test.height, test.width);
    }

    mnist_free(&test);
    return 0;
}
