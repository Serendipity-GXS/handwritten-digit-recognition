// main.c - program entry point
//
// Modes (added in later stages):
//   train   - train the network and save weights
//   eval    - load weights, report test accuracy / confusion matrix
//   demo    - load weights, predict a handwritten digit image
//
// Stage 0: minimal placeholder that verifies the toolchain works.

#include<stdio.h>
union
{
    short name;
    char age[2];
}li;

int main()
{
    li.name=1234; 
    printf("%x%x",li.age[0],li.age[1]);
    int n = sizeof(int);
    int c = sizeof(short);
    printf("\n%d,%d",n,c);
    return 0;
}