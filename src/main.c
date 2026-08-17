// main.c — temporary test entry for dense_forward.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "tensor.h"
#include "layer.h"
#include "model.h"
#include "data.h"

int main(void)
{
    srand(50);   //固定随机种子, 保证初始权重可复现
    MnistError Mnerr;
    ModelError merr;

    printf("=== Handwritten Digit Recognition ===");
    //载入训练集
    MnistSet Train;
    Mnerr = mnist_load(&Train,"data/train-images-idx3-ubyte","data/train-labels-idx1-ubyte");
    if(Mnerr != MNIST_OK){
        printf("\nMNIST_LOAD_ERROR:%d",Mnerr);
        return 1;
    }
    printf("\n[INFO] MNIST_Train load successfully.");

    //创建模型
    Model M;
    merr = model_init(&M,(const int[]){784,128,64,10,0});
    if(merr != MODEL_OK){
        printf("\nMODEL_INIT_ERROR:%d",merr);
    }
    printf("\n[INFO] Model init successfully.");

    //训练循环
    printf("\n[INFO] Train Start.");
    const int EPOCH = 10;
    const int PIXEL = 784;
    const int TRAIN_SIZE = Train.n;
    const float LR = 0.01;
    printf("\n[EPOCH:%d | PIXEL:%d | TRAIN_SIZE:%d | LR:%.4f]",EPOCH,PIXEL,TRAIN_SIZE,LR);

    Tensor input;
    tensor_create(&input,1,(const int[]){784});
    for(int epoch = 0;epoch < EPOCH;epoch ++){

        clock_t t0 = clock();
        float Loss = 0.0f;
        float acc = 0.0f;
        int correct = 0;

        for(int i = 0;i < TRAIN_SIZE;i ++){
            memcpy(input.data,&Train.images[i * PIXEL],PIXEL * sizeof(float));
            int label = Train.labels[i];
            //前向传播
            relu_forward(&input,&M.layer[0]);
            relu_forward(M.layer[0].activation_value,&M.layer[1]);
            dense_forward(M.layer[1].activation_value,&M.layer[2]);
            softmax_forward(&M.layer[2]);
            //计算损失函数
            Loss += softmax_crossentropy_loss(M.layer[2].activation_value,label);
            //判断预测正误
            int is_correct = 0;
            for(int j = 0;j < 10;j ++){
                if(M.layer[2].activation_value -> data[label] < M.layer[2].activation_value -> data[j]){
                    is_correct ++;
                }
            }
            if(!is_correct){
                correct ++;
            }
            //反向传播
            softmax_backward(&M.layer[2],label);
            dense_backward(M.layer[2].delta,&M.layer[2],M.layer[1].activation_value,M.layer[1].delta);
            relu_backward(&M.layer[1]);
            dense_backward(M.layer[1].delta,&M.layer[1],M.layer[0].activation_value,M.layer[0].delta);
            relu_backward(&M.layer[0]);
            dense_backward(M.layer[0].delta,&M.layer[0],&input,NULL);
            //参数更新
            sgd_update(&M.layer[0],LR);
            sgd_update(&M.layer[1],LR);
            sgd_update(&M.layer[2],LR);
        }
        double sec = (double)(clock() - t0) / CLOCKS_PER_SEC;
        acc = (float)correct / TRAIN_SIZE;
        Loss = Loss / TRAIN_SIZE;
        printf("\nepoch %02d | Loss:%.6f | acc:%.2f%% | time:%.3fs | (%d/%d)",epoch + 1,Loss,acc*100,sec,correct,TRAIN_SIZE);
    }

    //加载测试集合
    MnistSet Test;
    Mnerr = mnist_load(&Test,"data/t10k-images-idx3-ubyte","data/t10k-labels-idx1-ubyte");
    if(Mnerr != MNIST_OK){
        printf("\nMNIST_LOAD_ERROR:%d",Mnerr);
        return 1;
    }
    printf("\n[INFO] MNIST_Test load successfully.");
    printf("\n[INFO] Test start.");
    //测试循环
    clock_t t0 = clock();
    float Loss = 0.0f;
    float acc = 0.0f;
    int correct = 0;
    for(int i = 0;i < Test.n;i ++){
        memcpy(input.data,&Test.images[i * PIXEL],PIXEL * sizeof(float));
        int label = Test.labels[i];
        //前向传播
        relu_forward(&input,&M.layer[0]);
        relu_forward(M.layer[0].activation_value,&M.layer[1]);
        dense_forward(M.layer[1].activation_value,&M.layer[2]);
        softmax_forward(&M.layer[2]);
        //计算损失函数
        Loss += softmax_crossentropy_loss(M.layer[2].activation_value,label);
        //判断预测正误
        int is_correct = 0;
        for(int j = 0;j < 10;j ++){
            if(M.layer[2].activation_value -> data[label] < M.layer[2].activation_value -> data[j]){
                is_correct ++;
            }
        }
        if(!is_correct){
            correct ++;
        }        
    }
    Loss /= (float)Test.n;
    acc = correct / (float)Test.n;
    double sec = (double)(clock() - t0) / CLOCKS_PER_SEC;
    printf("\nTest result:");
    printf("\nacc:%.2f%% | avgLoss:%f | time:%fs | (%d/%d)",acc * 100,Loss,sec,correct,Test.n);

    //权重导出: 生成weights.h (main.exe从项目根运行, 故路径为src/weights.h)
    merr = model_save_weights(&M, "src/weights.h");
    if(merr != MODEL_OK){
        printf("\nSAVE_WEIGHTS_ERROR:%d",merr);
    }
    else{
        printf("\n[INFO] Weights saved to src/weights.h.");
    }

    //回读验证: 新建模型M2, 加载权重后重跑测试集, 准确率应与上面的M一致
    Model M2;
    merr = model_init(&M2,(const int[]){784,128,64,10,0});
    if(merr != MODEL_OK){
        printf("\nMODEL_INIT_ERROR:%d",merr);
    }
    model_load_weights(&M2);
    int correct2 = 0;
    for(int i = 0;i < Test.n;i ++){
        memcpy(input.data,&Test.images[i * PIXEL],PIXEL * sizeof(float));
        int label = Test.labels[i];
        relu_forward(&input,&M2.layer[0]);
        relu_forward(M2.layer[0].activation_value,&M2.layer[1]);
        dense_forward(M2.layer[1].activation_value,&M2.layer[2]);
        softmax_forward(&M2.layer[2]);
        int best = 0;   //argmax: 概率最大的类
        for(int j = 1;j < 10;j ++){
            if(M2.layer[2].activation_value->data[j] > M2.layer[2].activation_value->data[best])  best = j;
        }
        if(best == label)  correct2 ++;
    }
    printf("\n[INFO] Loaded model test: acc:%.2f%% | (%d/%d)",correct2 / (float)Test.n * 100,correct2,Test.n);
    model_free(&M2);


    model_free(&M);
    tensor_free(&input);
    mnist_free(&Train);
    mnist_free(&Test);
    return 0;
}
