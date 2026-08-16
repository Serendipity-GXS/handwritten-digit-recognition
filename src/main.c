// main.c — temporary test entry for dense_forward.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "tensor.h"
#include "layer.h"
#include "model.h"
#include "data.h"

int main(void)
{
    srand(42);   //固定随机种子, 保证初始权重可复现
    TensorError terr;
    LayerError lerr;
    ModelError merr;

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

    /* 9. 测试relu_backward: δz[i] = relu'(z[i]) · δa[i], 且actval只读不覆写 */
    /* 手算期望: actval={2,-1,0}, δa={0.5,-0.8,1.2}
       → 正的保留, 负的/零置0: δz={0.5, 0, 0}; actval 保持不变 */
    Tensor ract;
    terr = tensor_create(&ract, 1, (const int[]){3});
    if (terr != TENSOR_OK) { printf("create ract failed: %d\n", terr); return 1; }
    ract.data[0] = 2.0f;
    ract.data[1] = -1.0f;
    ract.data[2] = 0.0f;

    Tensor rdlt;
    terr = tensor_create(&rdlt, 1, (const int[]){3});
    if (terr != TENSOR_OK) { printf("create rdlt failed: %d\n", terr); return 1; }
    rdlt.data[0] = 0.5f;
    rdlt.data[1] = -0.8f;
    rdlt.data[2] = 1.2f;

    /* 组装隐藏层Layer: relu_backward只读activation_value和delta, 其余字段仅为完整组装 */
    Layer RL;
    RL.index = 0;
    RL.weight_col = 3;
    RL.weight_row = 2;
    RL.weight = &w;
    RL.bias = &b;
    RL.grad_weight = &gw;
    RL.grad_bias = &gb;
    RL.delta = &rdlt;
    RL.activation_value = &ract;

    lerr = relu_backward(&RL);
    printf("relu_backward return: %d\n", lerr);
    printf("delta_after  = {%.4f, %.4f, %.4f}\n", rdlt.data[0], rdlt.data[1], rdlt.data[2]);
    printf("actval_kept  = {%.4f, %.4f, %.4f}\n", ract.data[0], ract.data[1], ract.data[2]);

    tensor_free(&ract);
    tensor_free(&rdlt);

    /* 10. 测试sgd_update: w -= lr*gradW, b -= lr*gradB, 接着第8步算好的梯度 */
    /* lr = 0.01, 手算: weight[0]=0.5-0.01*(-1000)=10.5, weight[2]=0.2-0.01*1000=-9.8
       weight[5]=-0.5-0.01*(-1000)=9.5, bias[0]=0.1-0.01*(-1)=0.11, bias[1]=0.9-0.01*1=0.89 */
    lerr = sgd_update(&L, 0.01f);
    printf("sgd_update return: %d\n", lerr);
    printf("weight_after[0] = {%.4f, %.4f, %.4f}\n", w.data[0], w.data[1], w.data[2]);
    printf("weight_after[1] = {%.4f, %.4f, %.4f}\n", w.data[3], w.data[4], w.data[5]);
    printf("bias_after      = {%.4f, %.4f}\n", b.data[0], b.data[1]);

    tensor_free(&in);
    tensor_free(&w);
    tensor_free(&b);
    tensor_free(&act);
    tensor_free(&gw);
    tensor_free(&gb);
    tensor_free(&dlt);
    tensor_free(&pdelta);

    /* 11. 测试model_init/model_free: 一次建三层 */
    Model M;
    merr = model_init(&M, (const int[]){784, 128, 64, 10, 0});
    printf("model_init return: %d\n", merr);
    printf("layer_num(含输入层) = %d\n", M.layer_num);
    printf("model_shape = {%d, %d, %d, %d}\n",
           M.model_shape[0], M.model_shape[1], M.model_shape[2], M.model_shape[3]);
    for(int i = 0;i < M.layer_num - 1;i ++){
        printf("Layer%d: weight[%d][%d] size=%d, bias size=%d, delta size=%d, w[0]=%.6f\n",
               M.layer[i].index,
               M.layer[i].weight_row, M.layer[i].weight_col,
               M.layer[i].weight->size,
               M.layer[i].bias->size,
               M.layer[i].delta->size,
               M.layer[i].weight->data[0]);
    }
    model_free(&M);
    printf("model_free done\n");

    /* 12. 训练循环: 单样本SGD, 全训练集迭代EPOCHS轮 */
    {
        MnistSet train;
        MnistError nerr = mnist_load(&train,
            "data/train-images-idx3-ubyte",
            "data/train-labels-idx1-ubyte");
        if(nerr != MNIST_OK){
            printf("mnist_load failed: %d\n", nerr);
            return 1;
        }

        Model M;
        merr = model_init(&M, (const int[]){784, 128, 64, 10, 0});
        if(merr != MODEL_OK){
            printf("model_init failed: %d\n", merr);
            mnist_free(&train);
            return 1;
        }

        /* 输入缓冲只建一次, 每样本把784个像素拷进去 */
        Tensor input;
        terr = tensor_create(&input, 1, (const int[]){784});
        if(terr != TENSOR_OK){
            printf("create input failed: %d\n", terr);
            model_free(&M);
            mnist_free(&train);
            return 1;
        }

        const int EPOCHS = 10;
        const float LR = 0.01f;        //学习率, 0.1会触发ReLU死亡, 降到0.01
        const int PIXELS = 784;

        for(int epoch = 0;epoch < EPOCHS;epoch ++){
            float total_loss = 0.0f;
            int correct = 0;

            for(int i = 0;i < train.n;i ++){
                /* 第i张图拷入输入缓冲 */
                memcpy(input.data, &train.images[i * PIXELS], PIXELS * sizeof(float));
                int label = train.labels[i];

                /* 前向: 784→128(ReLU)→64(ReLU)→10(softmax) */
                relu_forward(&input, &M.layer[0]);
                relu_forward(M.layer[0].activation_value, &M.layer[1]);
                dense_forward(M.layer[1].activation_value, &M.layer[2]);  // z3 = W3·a2 + b3
                softmax_forward(&M.layer[2]);                             // p = softmax(z3)

                /* 损失 + 预测(argmax) */
                total_loss += softmax_crossentropy_loss(M.layer[2].activation_value, label);
                int pred = 0;
                for(int k = 1;k < 10;k ++){
                    if(M.layer[2].activation_value->data[k] >
                       M.layer[2].activation_value->data[pred])  pred = k;
                }
                if(pred == label)  correct ++;

                /* 反向: 从输出层逐层往输入层, 每层backward把误差写进上一层delta */
                softmax_backward(&M.layer[2], label);                    /* δ3 = p - y */
                dense_backward(M.layer[2].delta, &M.layer[2],
                               M.layer[1].activation_value, M.layer[1].delta);
                relu_backward(&M.layer[1]);                              /* δz2 = relu'(z2)·δa2 */
                dense_backward(M.layer[1].delta, &M.layer[1],
                               M.layer[0].activation_value, M.layer[0].delta);
                relu_backward(&M.layer[0]);
                dense_backward(M.layer[0].delta, &M.layer[0],
                               &input, NULL);                            /* 输入层不回传 */

                /* 参数更新: 三层权重/偏置各走一次SGD */
                sgd_update(&M.layer[0], LR);
                sgd_update(&M.layer[1], LR);
                sgd_update(&M.layer[2], LR);
            }

            printf("epoch %2d | avg_loss %.4f | acc %.2f%% (%d/%d)\n",
                   epoch + 1, total_loss / train.n,
                   (float)correct / train.n * 100.0f, correct, train.n);
        }

        /* 13. 测试集评估: 训练完成, 用t10k跑一遍前向看泛化能力 */
        {
            MnistSet test;
            nerr = mnist_load(&test,
                "data/t10k-images-idx3-ubyte",
                "data/t10k-labels-idx1-ubyte");
            if(nerr != MNIST_OK){
                printf("test mnist_load failed: %d\n", nerr);
                tensor_free(&input);
                model_free(&M);
                mnist_free(&train);
                return 1;
            }
            int correct = 0;
            for(int i = 0;i < test.n;i ++){
                memcpy(input.data, &test.images[i * PIXELS], PIXELS * sizeof(float));
                relu_forward(&input, &M.layer[0]);
                relu_forward(M.layer[0].activation_value, &M.layer[1]);
                dense_forward(M.layer[1].activation_value, &M.layer[2]);
                softmax_forward(&M.layer[2]);
                int pred = 0;
                for(int k = 1;k < 10;k ++){
                    if(M.layer[2].activation_value->data[k] >
                       M.layer[2].activation_value->data[pred])  pred = k;
                }
                if(pred == test.labels[i])  correct ++;
            }
            printf("test acc %.2f%% (%d/%d)\n",
                   (float)correct / test.n * 100.0f, correct, test.n);
            mnist_free(&test);
        }

        tensor_free(&input);
        model_free(&M);
        mnist_free(&train);
    }

    return 0;
}
