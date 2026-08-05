// main.c — temporary test entry for the data layer.
#include <stdio.h>
#include "data.h"
#include "tensor.h"

int main(void)
{
    int shape[4] = {2,3,4};
    Tensor t;
    TensorError err = tensor_create(&t,3,shape);
    if(err != TENSOR_OK){
        printf("TENSOR_ERROR:%d",err);
        return 1;
    }
    tensor_fill(&t,0.9);
    printf("shape0 - 3:");
    for(int i = 0;i < t.ndim; i ++){
        printf("%d ",t.shape[i]);
    }
    printf("\nsize:%d",t.size);
    printf("\ndata:");
    for(int i = 0;i < t.size; i ++){
        printf("%f ",t.data[i]);
    }
    printf("\nAll elements OK.");

    tensor_free(&t);
    return 0;
}
