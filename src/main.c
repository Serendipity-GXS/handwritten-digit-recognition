// main.c - program entry point
//
// Modes (added in later stages):
//   train   - train the network and save weights
//   eval    - load weights, report test accuracy / confusion matrix
//   demo    - load weights, predict a handwritten digit image
//
// Stage 0: minimal placeholder that verifies the toolchain works.

#include <stdio.h>

int main(void)
{
    printf("Hello, MNIST! (pure-C CNN project)\n");
    printf("Stage 0: toolchain + dataset ready.\n");
    return 0;
}
