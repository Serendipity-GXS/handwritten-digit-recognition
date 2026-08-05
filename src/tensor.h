#ifndef TENSOR_H
#define TENSOR_H

typedef struct {
    float *data;     // 连续内存,行优先
    int ndim;        // 维度个数,1~4
    int shape[4];    // 每维大小,如 {2,3,4}
    int size;        // 元素总数 = 各维乘积,如 2*3*4=24
} Tensor;

typedef enum{
    TENSOR_OK,
    TENSOR_NULL_POINTER_ERROR,
    TENSOR_MEMORY_ALLOCATION_FAILURE,
    TENSOR_INVALID_PARAMETER
}TensorError;

/**
  * @brief  创建 tensor 并分配内存
  * @param  t: 指向空 tensor 的指针
  * @param  ndim: tensor 维度,1~4
  * @param  shape: 各维大小数组
  * @retval TENSOR_OK 表示成功,否则返回对应错误码
  */
int tensor_create(Tensor *t, int ndim, const int shape[]);


/**
 * @brief tensor释放函数
 * @param t: 指向待释放的 tensor 指针
 */
void tensor_free(Tensor *t);

/**
 * @brief tensor填充函数
 * @param t: 指向待填充的 tensor 指针
 * @param value: 指定的填充值
 */
void tensor_fill(Tensor *t,float value);

#endif /*TENSOR_H*/