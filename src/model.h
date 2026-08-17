#ifndef MODEL_H
#define MODEL_H

#include "layer.h"

/* tensor池槽位布局: 每层固定 TENSORS_PER_LAYER 个tensor结构体, 按此顺序排列 */
enum{
    T_WEIGHT,           //权重矩阵
    T_BIAS,             //偏置向量
    T_GRAD_W,           //权重梯度
    T_GRAD_B,           //偏置梯度
    T_DELTA,            //误差缓冲
    T_ACT,              //激活值
    TENSORS_PER_LAYER   //每层tensor个数
};

typedef struct{
    int layer_num;      //模型层数(包含输入层), 如 {784,128,64,10} → 4
    int *model_shape;   //网络形状数组, 以0结尾, 由model_init拷贝、model_free释放
    Layer *layer;       //Dense层数组, 长度 layer_num-1 (输入层无权重矩阵)
    Tensor *tensor_pool;//所有tensor结构体, 数量 (layer_num-1)*TENSORS_PER_LAYER
}Model;

typedef enum{
    MODEL_OK,
    MODEL_NULL_POINTER_ERROR,
    MODEL_PARAMETER_INVALID,
    MODEL_MEMORY_ALLOCATION_FAILURE,
    MODEL_FILE_ERROR
}ModelError;

/**
 * @brief 模型初始化函数, 根据形状数组创建全部Dense层及其tensor
 * @param m 指向空Model结构体的指针
 * @param shape 网络形状数组, 如 {784,128,64,10,0}
 * @note shape必须以0结尾并预留足够长度(如定长int[16]), 函数遇0停止遍历
 * @note 输入层是纯向量, 无权重矩阵, 故Dense层数量 = 有效元素数-1
 * @note 权重/偏置均匀随机初始化[-limit,limit], limit=1/√(上一层神经元数); 梯度/误差清零
 * @retval MODEL_OK表示成功, 否则返回对应错误码
 */
int model_init(Model *m, const int shape[]);

/**
 * @brief 模型释放函数, 统一释放所有tensor、Layer数组与形状数组
 * @param m 指向待释放Model结构体的指针
 */
void model_free(Model *m);

/**
 * @brief 权重导出函数: 把模型每层权重/偏置写成C头文件
 * @param m 已训练完成的模型
 * @param path 输出文件路径, 如 "src/weights.h"
 * @note 生成W_0/B_0...等const float数组, 供model_load_weights回读或烧录嵌入式设备
 * @retval MODEL_OK表示成功, 否则返回对应错误码
 */
int model_save_weights(const Model *m, const char *path);

/**
 * @brief 权重回读函数: 从生成的weights.h把权重拷回模型
 * @param m 已由model_init建好结构的模型
 * @note 需#include "weights.h"; 网络层数变化时需同步增减LOAD_LAYER行数
 * @retval MODEL_OK表示成功, 否则返回对应错误码
 */
int model_load_weights(Model *m);

#endif /*MODEL_H*/
