#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "layer.h"

int dense_forward(const Tensor *pre_actval,Layer *next_layer){
    
    //参数校验
    if(pre_actval == NULL || next_layer == NULL){
        return LAYER_NULL_POINTER_ERROR;
    }
    if (pre_actval -> size != next_layer -> weight_col)
    {
        return LAYER_PARAMETER_MISMATCH;
    }

    int row = next_layer -> weight_row;
    int col = next_layer -> weight_col;
    float result = 0.0f;
    for(int i = 0;i < row;i ++){
        for(int j = 0;j < col;j ++){
            result += (next_layer -> weight -> data[i * col + j] * pre_actval -> data[j]);
        }
        result += (next_layer -> bias -> data[i]);
        next_layer -> activation_value -> data[i] = result;
        result = 0.0f;
    }
    return LAYER_OK;
}

int dense_backward(const Tensor *delta, Layer *this_layer, const Tensor *prev_actval, Tensor *prev_delta){
    //参数校验
    if(this_layer == NULL || delta == NULL || prev_actval == NULL){
         return LAYER_NULL_POINTER_ERROR;
    }

    int row = this_layer -> weight_row;
    int col = this_layer -> weight_col;

    //计算偏置的梯度，其实和∂L/∂z是相同的,直接复制
    for(int i = 0;i < row;i ++){
        this_layer -> grad_bias -> data[i] = delta -> data[i];
    }
    //计算权重的梯度: 外积 grad_weight[i][j] = delta[i] * prev_actval[j]
    for(int i = 0;i < row;i ++){
        for(int j = 0;j < col;j ++){
            this_layer -> grad_weight -> data[i * col + j] = delta -> data[i] * prev_actval -> data[j];
        }
    }
    //回传信号: δ_prev[j] = Σ_i weight[i][j] * delta[i]，写入上一层误差缓冲prev_delta
    //输入层(prev_delta == NULL)时不需要回传，只算梯度
    if(prev_delta != NULL){
        for(int j = 0;j < col;j ++){
            float sum = 0.0f;
            for(int i = 0;i < row;i ++){
                sum += this_layer -> weight -> data[i * col + j] * delta -> data[i];
            }
            prev_delta -> data[j] = sum;
        }
    }
    return LAYER_OK;
}

int relu_activation(Layer *layer_to_be_activated){
    //参数校验
    if(layer_to_be_activated == NULL ||
       layer_to_be_activated -> activation_value == NULL ||
       layer_to_be_activated -> weight == NULL ||
       layer_to_be_activated -> bias == NULL){
    return LAYER_NULL_POINTER_ERROR; //防止被错误调用，激活一个空Layer
    }

    //激活dense_forward返回的结果
    int actval_size = layer_to_be_activated -> activation_value -> size;
    for(int i = 0;i < actval_size;i ++){
        if(layer_to_be_activated -> activation_value -> data[i] < 0){
            layer_to_be_activated -> activation_value -> data[i] = 0;
        }
    }

    return LAYER_OK;
}

int relu_forward(const Tensor *pre_actval,Layer *next_layer){
    LayerError err = dense_forward(pre_actval,next_layer);
    if(err != LAYER_OK) return err;
    err = relu_activation(next_layer);
    return err;
}

int softmax_forward(Layer *layer_to_be_softmax){
    //参数校验
    if(layer_to_be_softmax == NULL ||
       layer_to_be_softmax -> activation_value == NULL ){
    return LAYER_NULL_POINTER_ERROR; //防止被错误调用，操作一个空Layer
    }

    int size = layer_to_be_softmax -> activation_value -> size;
    float max = layer_to_be_softmax -> activation_value -> data[0];
    float sum = 0.0f;

    for(int j = 0;j < size;j ++){
        if(max < layer_to_be_softmax -> activation_value -> data[j]){
            max = layer_to_be_softmax -> activation_value -> data[j];
        }
    }

    for(int i = 0;i < size;i ++){
        layer_to_be_softmax -> activation_value -> data[i] = exp(layer_to_be_softmax -> activation_value -> data[i] - max);
        sum += layer_to_be_softmax -> activation_value -> data[i];
    }

    for(int k = 0;k < size;k ++){
        layer_to_be_softmax -> activation_value -> data[k] /= sum;
    }

    return LAYER_OK;
}

int softmax_backward(Layer *layer_to_be_backward,int label){
    //参数校验
    if(layer_to_be_backward == NULL ||
       layer_to_be_backward -> activation_value == NULL ||
       layer_to_be_backward -> delta == NULL ){
    return LAYER_NULL_POINTER_ERROR; //防止被错误调用，操作一个空Layer
    }
    assert(label < layer_to_be_backward -> activation_value -> size && label >= 0);

    //δ = p - y: 先把概率p拷贝到delta缓冲, 再在label处减1
    for(int i = 0;i < layer_to_be_backward -> activation_value -> size;i ++){
        layer_to_be_backward -> delta -> data[i] = layer_to_be_backward -> activation_value -> data[i];
    }
    layer_to_be_backward -> delta -> data[label] -= 1.0f;
    return LAYER_OK;
}

float softmax_crossentropy_loss(const Tensor *probs, int label){
    //参数校验
    assert(probs != NULL && label < probs -> size && label >= 0);
    return - log(probs -> data[label] + 0.000001);
    //加一个0.000001是为了防止softmax计算后得到0后直接带入log计算导致错误
}