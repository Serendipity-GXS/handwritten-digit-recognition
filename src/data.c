// data.c — MNIST dataset loader (idx format) implementation.
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include "data.h"
/*========================== Tool Function ==========================*/

//数据拼接函数
static int bin_to_int(unsigned char* b){
    assert(b != NULL);
    return (b[0] << 24) | (b[1] << 16) | (b[2] << 8) | b[3];
}

/*========================== Functional Function ==========================*/

// 读取并载入数据集
int mnist_load(MnistSet *set, const char *img_path, const char *lbl_path)
{
    int err = MNIST_OK;             //统一错误码，在cleanup里统一返回
    int success = 0;                //成功标志：数据读取完成则所有权移交调用者
    FILE *img = NULL;               //创建指向图像集的指针
    FILE *lbl = NULL;               //创建指向标签集的指针
    int prefix_img,prefix_lbl;
    int n,row,col;
    unsigned char b1[4];
    unsigned char b2[4];
    unsigned char pixel;
    unsigned char label;

    //参数校验，此时还没占用任何资源，直接返回即可
    if(set == NULL || img_path == NULL || lbl_path == NULL){
        return MNIST_INVALID_PARAMETER;
    }

    //malloc前先置空，cleanup里free(NULL)是安全的
    set -> images = NULL;           
    set -> labels = NULL;

    //尝试打开文件
    img = fopen(img_path,"rb");
    lbl = fopen(lbl_path,"rb");
    if(img == NULL || lbl == NULL){
        err = MNIST_FILE_PATH_ERROR;
        goto cleanup;
    }

    //校验magic number
    if(fread(b1,1,4,img) != 4 || fread(b2,1,4,lbl) != 4){
        err = MNIST_FILE_CORRUPTION;
        goto cleanup;
    }
    prefix_img = bin_to_int(b1);
    prefix_lbl = bin_to_int(b2);
    if(prefix_img != 0x0803 || prefix_lbl != 0x0801){
        err = MNIST_FORMAT_ERROR;
        goto cleanup;
    }

    //读取image结构数据
    if(fread(b1,1,4,img) != 4){
        err = MNIST_FILE_CORRUPTION;
        goto cleanup;
    }
    n = bin_to_int(b1);

    if(fread(b1,1,4,img) != 4 || fread(b2,1,4,img) != 4){
        err = MNIST_FILE_CORRUPTION;
        goto cleanup;
    }
    row = bin_to_int(b1);
    col = bin_to_int(b2);

    //标签文件也有自己的count，必须读掉，否则数据区读取会整体错位4字节
    if(fread(b2,1,4,lbl) != 4){
        err = MNIST_FILE_CORRUPTION;
        goto cleanup;
    }
    int n_lbl = bin_to_int(b2);
    if(n_lbl != n){ //校验图像集和标签集数据数量是否一致
        err = MNIST_FORMAT_ERROR;
        goto cleanup;
    }

    //载入数据集
    set -> n = n;
    set -> height = row;
    set -> width = col;

    set -> images = (float*)malloc(n*col*row*sizeof(float));
    set -> labels = (unsigned char*)malloc(n*sizeof(unsigned char));
    if(set -> images == NULL || set -> labels == NULL){
        err = MNIST_MEMORY_ALLOCATION_FAILURE;
        goto cleanup;
    }

    for(int i = 0;i < n;i++){
        for(int j = 0;j < row;j++){
            for(int k = 0;k < col;k++){
                if(fread(&pixel,1,1,img) != 1){
                    err = MNIST_FILE_CORRUPTION;
                    goto cleanup;
                }
                set -> images[i * col * row + j * col + k] = pixel / 255.0f;
            }
        }
    }

    for(int i = 0;i < n;i++){
        if(fread(&label,1,1,lbl) != 1){
            err = MNIST_FILE_CORRUPTION;
            goto cleanup;
        }
        set -> labels[i] = label;
    }

    success = 1;    //数据读取完毕，所有权移交调用者，cleanup 不再释放

cleanup:
    if(img)  fclose(img);
    if(lbl)  fclose(lbl);
    if(!success){               //只有失败才释放数据，成功时数据归调用者
        free(set -> images);
        free(set -> labels);
    }
    return err;
}

//释放数据集占用的内存空间
void mnist_free(MnistSet *set){
    if(set == NULL){
        return;
    }
    free(set -> images);
    free(set -> labels);
    set -> images = NULL;
    set -> labels = NULL;
}

// Prints one image (h*w floats) as ASCII art, for visual verification.
void print_ascii_image(const float *img, int h, int w){
    static const char ramp[] = "..:-=+*#%@";
    for(int j = 0;j < h;j ++){
        for(int k = 0;k < w;k ++){
            float pixel = img[j * w + k]; 
            int idx = (int)(pixel * 9.0f);
            putchar(ramp[idx]);
            putchar(' ');
        }
        putchar('\n');
    }
}