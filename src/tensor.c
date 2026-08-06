#include <stdio.h>
#include <stdlib.h>
#include "tensor.h"


/*========================== Functional Function ==========================*/

//初始化创建一个tensor
int tensor_create(Tensor *t, int ndim, const int shape[]){

    int size = 1;
    TensorError err = TENSOR_OK;

    //参数检查
    if(t == NULL || shape == NULL){
        err = TENSOR_NULL_POINTER_ERROR;
        goto cleanup;
    }

    t -> data = NULL;   //使用前置空

    if(ndim <= 0){
        err = TENSOR_INVALID_PARAMETER;
        goto cleanup;
    }
    for(int i = 0;i < ndim;i ++){
        if(shape[i] <= 0){
            err = TENSOR_INVALID_PARAMETER;
            goto cleanup;
        }
        size *= shape[i];
    }

    //复制tensor属性参数
    t -> ndim = ndim;
    t -> size = size;
    for(int j = 0;j < ndim; j++){
        t -> shape[j] = shape[j];
    }

    //分配内存空间
    t -> data = (float*)malloc(size * sizeof(float));
    if(t -> data == NULL){
        err = TENSOR_MEMORY_ALLOCATION_FAILURE;
        goto cleanup;
    }
    cleanup:
        return err;
}

//tensor释放函数
void tensor_free(Tensor *t){
    //参数校验
    if(t == NULL){
        return;
    }
    free(t -> data);
    t -> data = NULL; //置空，防止double free报错
}


//tensor填充函数
void tensor_fill(Tensor *t,float value){
    //参数检查
    if(t == NULL){
        return;
    }
    for(int i = 0;i < t -> size;i ++){
        t -> data[i] = value;
    }
}

