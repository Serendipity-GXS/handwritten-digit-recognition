// main.c — temporary test entry for dense_forward.
#include <stdio.h>
#include "tensor.h"
#include "layer.h"

int main(void)
{
    TensorError terr;
    LayerError lerr;

    /* 1. 输入: 3 个元素 {1000, 999, -1000} */
    Tensor in;
    terr = tensor_create(&in, 1, (const int[]){3});
    if (terr != TENSOR_OK) { printf("create in failed: %d\n", terr); return 1; }
    in.data[0] = 1000.0f;
    in.data[1] = 999.0f;
    in.data[2] = -1000.0f;

    /* 2. 权重: [2][3], data = {0.5, -1.0, 0.2, 1.0, 0.0, -0.5} */
    Tensor w;
    terr = tensor_create(&w, 2, (const int[]){2, 3});
    if (terr != TENSOR_OK) { printf("create w failed: %d\n", terr); return 1; }
    float wv[] = {0.5f, -1.0f, 0.2f, 1.0f, 0.0f, -0.5f};
    for (int i = 0; i < 6; i++) w.data[i] = wv[i];

    /* 3. 偏置: [2] = {0.1, 0.9} */
    Tensor b;
    terr = tensor_create(&b, 1, (const int[]){2});
    if (terr != TENSOR_OK) { printf("create b failed: %d\n", terr); return 1; }
    b.data[0] = 0.1f;
    b.data[1] = 0.9f;

    /* 4. 输出: [2] */
    Tensor act;
    terr = tensor_create(&act, 1, (const int[]){2});
    if (terr != TENSOR_OK) { printf("create act failed: %d\n", terr); return 1; }

    /* 4.5 梯度: 与 weight/bias 同形状, 清零 */
    Tensor gw;
    terr = tensor_create(&gw, 2, (const int[]){2, 3});
    if (terr != TENSOR_OK) { printf("create gw failed: %d\n", terr); return 1; }
    tensor_fill(&gw, 0.0f);

    Tensor gb;
    terr = tensor_create(&gb, 1, (const int[]){2});
    if (terr != TENSOR_OK) { printf("create gb failed: %d\n", terr); return 1; }
    tensor_fill(&gb, 0.0f);

    /* 4.6 误差缓冲: 与本层神经元数同形状, 清零 */
    Tensor dlt;
    terr = tensor_create(&dlt, 1, (const int[]){2});
    if (terr != TENSOR_OK) { printf("create dlt failed: %d\n", terr); return 1; }
    tensor_fill(&dlt, 0.0f);

    /* 5. 组装 Layer */
    Layer L;
    L.index = 1;
    L.weight_col = 3;
    L.weight_row = 2;
    L.weight = &w;
    L.bias = &b;
    L.grad_weight = &gw;
    L.grad_bias = &gb;
    L.delta = &dlt;
    L.activation_value = &act;

    /* 6. 前向 */
    lerr = relu_forward(&in, &L);
    printf("dense_forward return: %d\n", lerr);
    printf("activation[0] = %.4f\n", act.data[0]);
    printf("activation[1] = %.4f\n", act.data[1]);

    /*测试softmax*/
    lerr = softmax_forward(&L);
    printf("softmax_forward return: %d\n", lerr);
    printf("activation_softmaxed[0] = %.4f\n", act.data[0]);
    printf("activation_softmaxed[1] = %.4f\n", act.data[1]);

    /* 7. 测试反向: softmax_backward 把概率p变成误差δ = p - y, 写入delta缓冲 */
    lerr = softmax_backward(&L, 0);   /* label = 0, 则 δ = {p0-1, p1} */
    printf("softmax_backward return: %d\n", lerr);
    printf("delta       = {%.4f, %.4f}\n", dlt.data[0], dlt.data[1]);
    printf("actval(p)   = {%.4f, %.4f}\n", act.data[0], act.data[1]);

    /* 8. 测试dense_backward: 三个产物 grad_weight / grad_bias / 回传信号 */
    Tensor pdelta;
    terr = tensor_create(&pdelta, 1, (const int[]){3});
    if (terr != TENSOR_OK) { printf("create pdelta failed: %d\n", terr); return 1; }
    tensor_fill(&pdelta, 0.0f);

    lerr = dense_backward(&dlt, &L, &in, &pdelta);
    printf("dense_backward return: %d\n", lerr);
    printf("grad_weight[0] = {%.4f, %.4f, %.4f}\n", gw.data[0], gw.data[1], gw.data[2]);
    printf("grad_weight[1] = {%.4f, %.4f, %.4f}\n", gw.data[3], gw.data[4], gw.data[5]);
    printf("grad_bias     = {%.4f, %.4f}\n", gb.data[0], gb.data[1]);
    printf("prev_delta    = {%.4f, %.4f, %.4f}\n", pdelta.data[0], pdelta.data[1], pdelta.data[2]);

    tensor_free(&in);
    tensor_free(&w);
    tensor_free(&b);
    tensor_free(&act);
    tensor_free(&gw);
    tensor_free(&gb);
    tensor_free(&dlt);
    tensor_free(&pdelta);

    return 0;
}
