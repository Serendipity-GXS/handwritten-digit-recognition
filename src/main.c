#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "tensor.h"
#include "layer.h"
#include "model.h"
#include "data.h"

//这里的宏定义用于模式切换。MODE_TRAIN为训练模式(训练模型并导出weights.h)
//把宏定义注释掉就是推理模式，直接加载训练好的weights.h，跳过训练直接测试
#define MODE_TRAIN

int main(void)
{
    MnistError Mnerr;
    ModelError merr;
    const int PIXEL = 784;

    Tensor input;   //输入缓冲, 训练/测试共用
    tensor_create(&input,1,(const int[]){784});

    printf("=== Handwritten Digit Recognition ===");

#ifdef MODE_TRAIN
    printf("\n[MODE] Train mode on.");
    srand(50);   //固定随机种子, 保证初始权重可复现

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
        return 1;
    }
    printf("\n[INFO] Model init successfully.");

    //训练循环
    printf("\n[INFO] Train Start.");
    const int EPOCH = 10;           //训练轮次
    const int TRAIN_SIZE = Train.n; //控制训练样本数量，默认全量训练
    const float LR = 0.01;          //学习率（不要超过0.1，训练会不收敛！不过也可以改着玩玩哈哈）
    printf("\n[EPOCH:%d | PIXEL:%d | TRAIN_SIZE:%d | LR:%.4f]",EPOCH,PIXEL,TRAIN_SIZE,LR);

    for(int epoch = 0;epoch < EPOCH;epoch ++){

        clock_t t0 = clock();
        float Loss = 0.0f;  //每轮训练后的平均损失
        float acc = 0.0f;   //每轮训练中的准确率
        int correct = 0;    //成功识别样本计数

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
        printf("\nepoch %02d | avgLoss:%.6f | acc:%.2f%% | time:%.3fs | (%d/%d)",epoch + 1,Loss,acc*100,sec,correct,TRAIN_SIZE);
    }

    //权重导出: 生成weights.h (main.exe从项目根运行, 故路径为src/weights.h)
    merr = model_save_weights(&M, "src/weights.h");
    if(merr != MODEL_OK){
        printf("\nSAVE_WEIGHTS_ERROR:%d",merr);
    }
    else{
        printf("\n[INFO] Weights saved to src/weights.h.");
    }

    mnist_free(&Train);

#else

    printf("\n[MODE] infer mode on.");
    //推理模式,直接加载已导出的weights.h
    Model M;
    merr = model_init(&M,(const int[]){784,128,64,10,0});
    if(merr != MODEL_OK){
        printf("\nMODEL_INIT_ERROR:%d",merr);
        return 1;
    }
    printf("\n[INFO] Model init successfully.");

    merr = model_load_weights(&M);
    if(merr != MODEL_OK){
        printf("\nLOAD_WEIGHTS_ERROR:%d",merr);
        return 1;
    }
    printf("\n[INFO] Weights loaded from src/weights.h.");

#endif

    //后面这段训练/推理两种模式共用(验证测试集)
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
    const int WRONG_SHOW = 8;   //最多展示的识别错误样本数
    int wrong_idx[WRONG_SHOW];  //记录识别错误样本在测试集中的索引
    int wrong_cnt = 0;
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
        else if(wrong_cnt < WRONG_SHOW){   //识别错误: 记录前几个样本索引用于后面展示
            wrong_idx[wrong_cnt ++] = i;
        }
    }
    Loss /= (float)Test.n;
    acc = correct / (float)Test.n;
    double sec = (double)(clock() - t0) / CLOCKS_PER_SEC;
    printf("\nTest result:");
    printf("\nacc:%.2f%% | avgLoss:%f | time:%fs | (%d/%d)",acc * 100,Loss,sec,correct,Test.n);

    //展示前几个识别错误的样本
    if(wrong_cnt > 0){
        printf("\n\n==== Misclassified samples (show %d, total wrong = %d) ====", wrong_cnt, Test.n - correct);
        for(int k = 0;k < wrong_cnt;k ++){
            int i = wrong_idx[k];
            int label = Test.labels[i];
            memcpy(input.data,&Test.images[i * PIXEL],PIXEL * sizeof(float));
            //重新跑一下前向传播，计算模型输出的概率分布
            relu_forward(&input,&M.layer[0]);
            relu_forward(M.layer[0].activation_value,&M.layer[1]);
            dense_forward(M.layer[1].activation_value,&M.layer[2]);
            softmax_forward(&M.layer[2]);
            const Tensor *probs = M.layer[2].activation_value;  //这里用const声明一下，不要把输出的结果改掉了
            //模型最终判断
            int predict = 0;
            for(int j = 1;j < 10;j ++){
                if(probs -> data[j] > probs -> data[predict])  predict = j;
            }

            printf("\n\n----- Sample #%d | true label: %d | predicted: %d (WRONG) -----\n",i, label, predict);
            print_ascii_image(&Test.images[i * PIXEL], Test.height, Test.width);
            
            //打印模型输出的概率分布
            printf("prob: ");
            for(int row = 0;row < 2;row ++){
                if(row > 0)  printf("      ");   //第二行缩进, 和第一行的数据对整齐(ヾ(≧▽≦*)o)
                for(int col = 0;col < 5;col ++){
                    int j = row * 5 + col;
                    printf(" [%d]:%.4f%s ", j, probs -> data[j], (j == predict) ? "*" : " ");
                    //这里用*注明模型认为的图像显示的数字
                }
                printf("\n");
            }
        }
    }

    model_free(&M);
    tensor_free(&input);
    mnist_free(&Test);
    return 0;
}
