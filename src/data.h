// data.h — MNIST dataset loader (idx format).
// Loads two idx files into a MnistSet, and renders one image as ASCII art.
#ifndef DATA_H
#define DATA_H

typedef struct {
    int n;                    // number of samples
    int height, width;        // 28, 28
    float *images;            // [n][height*width], normalized to [0,1]
    unsigned char *labels;    // [n], 0~9
} MnistSet;

typedef enum{
    MNIST_OK,
    MNIST_FORMAT_ERROR,
    MNIST_FILE_PATH_ERROR,
    MNIST_NULL_POINTER_ERROR,
    MNIST_FILE_CORRUPTION,
    MNIST_MEMORY_ALLOCATION_FAILURE,
    MNIST_INVALID_PARAMETER
}MnistError;

// Reads the image and label idx files into set. Returns 0 on success, nonzero on failure.
int  mnist_load(MnistSet *set, const char *img_path, const char *lbl_path);

// Frees memory owned by set.
void mnist_free(MnistSet *set);

// Prints one image (h*w floats) as ASCII art, for visual verification.
void print_ascii_image(const float *img, int h, int w);

#endif /* DATA_H */
