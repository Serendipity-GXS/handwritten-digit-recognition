#ifndef LAYER_H
#define LAYER_H

#include <stdio.h>
#include <math.h>
#include "tensor.h"

typedef struct{
    int index;  //layer层级
    int weight_col; //权重矩阵列数
    int weight_row; //权重矩阵行数
    Tensor *weight; //权重参数
    Tensor *bias;   //偏置参数
    Tensor *grad_weight;    //权重梯度
    Tensor *grad_bias;  //偏置梯度
    Tensor *activation_value;   //本层的激活值
    Tensor *delta;  //本层的误差缓冲(反向传播用,存对z的梯度)
}Layer;

typedef enum{
    LAYER_OK,
    LAYER_NULL_POINTER_ERROR,
    LAYER_PARAMETER_MISMATCH,
    LAYER_PARAMETER_INVALID,
    LAYER_MEMORY_ALLOCATION_FAILURE
}LayerError;

/**
 * @brief 前向传播函数
 * @param pre_actval 前一层神经元的激活值
 * @param next_layer 后一层神经元，layer包含权重、偏置等信息
 * @retval 返回函数的执行状态。前向传播结果存储在layer -> activation_value 中
 */
int dense_forward(const Tensor *pre_actval,Layer *next_layer);

/**
 * @brief dense层反向传播函数，计算本层权重/偏置梯度，并向上一层回传误差信号
 * @param delta 本层神经元误差信号，即损失函数对加权求和结果z的偏导数 ∂L/∂z
 * @param this_layer 本层Layer，梯度结果写入其grad_weight与grad_bias字段
 * @param prev_actval 上一层神经元的激活值，用于计算权重梯度
 * @param prev_delta 上一层神经元的误差缓冲，接收回传信号；输入层(无上一层Layer)传NULL
 * @note 权重梯度 = delta ⊗ prev_actval(外积)，偏置梯度 = delta
 * @note 回传信号 δ_prev[j] = Σ_i weight[i][j]·delta[i]，写入prev_delta，供上一层relu_backward使用
 * @retval 返回函数的执行状态。
 */
int dense_backward(const Tensor *delta, Layer *this_layer, const Tensor *prev_actval, Tensor *prev_delta);

/**
 * @brief ReLu激活层，本身不具备前向传播功能，需要搭配dense_forward使用
 * @param layer_to_be_activated 指向需要激活的神经元层的指针
 * @retval 返回函数的执行状态。
 */
int relu_activation(Layer *layer_to_be_activated);

/**
 * @brief 前向传播函数，自带ReLu激活层
 * @param pre_actval 前一层神经元的激活值
 * @param next_layer 后一层神经元，layer包含权重、偏置等信息
 * @retval 返回函数的执行状态。前向传播并激活后的结果存储在layer -> activation_value 中
 */
int relu_forward(const Tensor *pre_actval,Layer *next_layer);

/**
 * @brief softmax函数，将输出层评分转化为合理的概率分布
 * @note softmax操作是就地进行的。如果需要原始评分数据请在执行函数前备份
 * @param layer_to_be_softmax 指向需执行softmax操作的神经元层的指针
 * @retval 返回函数的执行状态。
 */
int softmax_forward(Layer *layer_to_be_softmax);

/**
 * @brief backward函数，计算输出层误差 δ = p - y
 * @param layer_to_be_backward 指向需执行backward操作的神经元层的指针
 * @param label 当前期望结果标签
 * @note 计算结果写入layer->delta缓冲；activation_value(概率p)保持不变
 * @retval 返回函数的执行状态。
 */
int softmax_backward(Layer *layer_to_be_backward,int label);

/**
 * @brief 交叉熵损失计算函数
 * @param probs 指向softmax操作后的输出层activation_value字段的指针
 * @param label 标签值，等于正确答案
 * @note 函数内部不对label作特别校验，调用者需确保传参正确(0 - 9)
 * @retval 返回计算得到的交叉熵损失，浮点值
 */
float softmax_crossentropy_loss(const Tensor *probs, int label);


#endif /*LAYER_H*/