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

/**
 * @brief MNIST数据加载函数，读取图片和标签两个idx文件
 * @param set 指向MnistSet结构体的指针,加载的数据写入其中
 * @param img_path 图片idx文件的路径
 * @param lbl_path 标签idx文件的路径
 * @retval MNIST_OK表示成功,否则返回对应错误码
 */
int  mnist_load(MnistSet *set, const char *img_path, const char *lbl_path);

/**
 * @brief MNIST数据集释放函数
 * @param set 指向待释放的MnistSet结构体的指针
 */
void mnist_free(MnistSet *set);

/**
 * @brief 图片字符画打印函数,用于可视化验证
 * @param img 图片像素数据, h*w个浮点数, 已归一化到[0,1]
 * @param h 图片高度
 * @param w 图片宽度
 */
void print_ascii_image(const float *img, int h, int w);

#endif /* DATA_H */
