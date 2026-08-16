#include <stdlib.h>
#include <math.h>
#include "model.h"

/* 统计形状数组有效元素个数, 遇到结尾0停止 */
static int count_shape(const int shape[]){
    int n = 0;
    while(shape[n] != 0)  n++;
    return n;
}

int model_init(Model *m, const int shape[]){
    //参数校验
    if(m == NULL || shape == NULL)  return MODEL_NULL_POINTER_ERROR;

    //先置空, 保证出错时清理路径安全
    m->layer_num = 0;
    m->model_shape = NULL;
    m->layer = NULL;
    m->tensor_pool = NULL;

    int layer_num = count_shape(shape);
    if(layer_num < 2)  return MODEL_PARAMETER_INVALID;   //模型至少包含输入+输出两层

    //各层神经元数必须为正
    for(int i = 0;i < layer_num;i ++){
        if(shape[i] <= 0)  return MODEL_PARAMETER_INVALID;
    }
    m -> layer_num = layer_num;

    //拷贝形状数组(含结尾0), 由model_free统一释放
    m -> model_shape = (int*)malloc((layer_num + 1) * sizeof(int));
    if(m -> model_shape == NULL)  return MODEL_MEMORY_ALLOCATION_FAILURE;
    for(int i = 0;i <= layer_num;i ++){
        m->model_shape[i] = shape[i];
    }

    int dense_num = layer_num - 1;

    //分配Dense层数组
    m->layer = (Layer*)calloc(dense_num, sizeof(Layer));
    if(m->layer == NULL){
        free(m->model_shape);
        m->model_shape = NULL;
        return MODEL_MEMORY_ALLOCATION_FAILURE;
    }

    //分配tensor池, calloc保证所有data指针初始为NULL, 出错清理时tensor_free安全
    m->tensor_pool = (Tensor*)calloc(dense_num * TENSORS_PER_LAYER, sizeof(Tensor));
    if(m->tensor_pool == NULL){
        free(m->layer);
        m->layer = NULL;
        free(m->model_shape);
        m->model_shape = NULL;
        return MODEL_MEMORY_ALLOCATION_FAILURE;
    }

    //逐层创建tensor并初始化
    for(int i = 0;i < dense_num;i ++){
        Layer *l = &m->layer[i];
        Tensor *t = &m->tensor_pool[i * TENSORS_PER_LAYER];
        int row = shape[i + 1];   //本层神经元数
        int col = shape[i];       //上一层神经元数

        l->index = i + 1;
        l->weight_row = row;
        l->weight_col = col;
        l->weight         = &t[T_WEIGHT];
        l->bias           = &t[T_BIAS];
        l->grad_weight    = &t[T_GRAD_W];
        l->grad_bias      = &t[T_GRAD_B];
        l->delta          = &t[T_DELTA];
        l->activation_value = &t[T_ACT];

        if(tensor_create(l->weight, 2, (const int[]){row, col}) != TENSOR_OK)  goto cleanup_tensor;
        if(tensor_create(l->bias, 1, (const int[]){row}) != TENSOR_OK)         goto cleanup_tensor;
        if(tensor_create(l->grad_weight, 2, (const int[]){row, col}) != TENSOR_OK)  goto cleanup_tensor;
        if(tensor_create(l->grad_bias, 1, (const int[]){row}) != TENSOR_OK)         goto cleanup_tensor;
        if(tensor_create(l->delta, 1, (const int[]){row}) != TENSOR_OK)             goto cleanup_tensor;
        if(tensor_create(l->activation_value, 1, (const int[]){row}) != TENSOR_OK)  goto cleanup_tensor;

        //初始化: 权重/偏置随机, 梯度/误差清零; 激活值由前向计算产生, 无需初始化
        float limit = 1.0f / sqrtf((float)col);
        tensor_fill_random(l->weight, limit);
        tensor_fill_random(l->bias, limit);
        tensor_fill(l->grad_weight, 0.0f);
        tensor_fill(l->grad_bias, 0.0f);
        tensor_fill(l->delta, 0.0f);
    }
    return MODEL_OK;

cleanup_tensor:
    model_free(m);   //池内未创建成功的tensor data为NULL, tensor_free对NULL安全
    return MODEL_MEMORY_ALLOCATION_FAILURE;
}

void model_free(Model *m){
    if(m == NULL)  return;
    //释放tensor池中每个tensor的data
    if(m->tensor_pool != NULL){
        int total = (m->layer_num - 1) * TENSORS_PER_LAYER;
        for(int i = 0;i < total;i ++){
            tensor_free(&m->tensor_pool[i]);
        }
        free(m->tensor_pool);
        m->tensor_pool = NULL;
    }
    free(m->layer);
    m->layer = NULL;
    free(m->model_shape);
    m->model_shape = NULL;
}
