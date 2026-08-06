// main.c — temporary test entry for the data layer.
#include <stdio.h>
#include "data.h"
#include "tensor.h"

int main(void)
{
    float x = 200.0f;  //输入
    float y = 600.0f;  //期望
    float w = 0.0f;  //初值
    float n = 0.00001f;  //学习率

    for(int i = 0;i < 10;i ++){
        float y_hat = w * x;                 // 前向:预测
        float L = (y_hat - y) * (y_hat - y); // 损失
        float dLdw = 2 * (y_hat - y) * x;    // 梯度
        w = w - n * dLdw;                  // 更新
        printf("iter %d: w=%.6f L=%.6f\n", i, w, L);
    }

    return 0;
}
