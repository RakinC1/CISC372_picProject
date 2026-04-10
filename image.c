#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include <string.h>
#include <stdlib.h>
#include <pthread.h>
#include "image.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

Matrix algorithms[]={
    {{0,-1,0},{-1,4,-1},{0,-1,0}},
    {{0,-1,0},{-1,5,-1},{0,-1,0}},
    {{1/9.0,1/9.0,1/9.0},{1/9.0,1/9.0,1/9.0},{1/9.0,1/9.0,1/9.0}},
    {{1.0/16,1.0/8,1.0/16},{1.0/8,1.0/4,1.0/8},{1.0/16,1.0/8,1.0/16}},
    {{-2,-1,0},{-1,1,1},{0,1,2}},
    {{0,0,0},{0,1,0},{0,0,0}}
};


typedef struct {
    Image* srcImage;
    Image* destImage;
    Matrix algorithm;
    int start_row;
    int end_row;
} ThreadData;

void* thread_func(void* arg){
    ThreadData* data = (ThreadData*)arg;

    for (int row = data->start_row; row < data->end_row; row++){
        for (int pix = 0; pix < data->srcImage->width; pix++){
            for (int bit = 0; bit < data->srcImage->bpp; bit++){
                data->destImage->data[Index(pix,row,data->srcImage->width,bit,data->srcImage->bpp)] =
                    getPixelValue(data->srcImage,pix,row,bit,data->algorithm);
            }
        }
    }
    return NULL;
}


uint8_t getPixelValue(Image* srcImage,int x,int y,int bit,Matrix algorithm){
    int px,mx,py,my;
    px=x+1; py=y+1; mx=x-1; my=y-1;
    if (mx<0) mx=0;
    if (my<0) my=0;
    if (px>=srcImage->width) px=srcImage->width-1;
    if (py>=srcImage->height) py=srcImage->height-1;

    uint8_t result=
        algorithm[0][0]*srcImage->data[Index(mx,my,srcImage->width,bit,srcImage->bpp)]+
        algorithm[0][1]*srcImage->data[Index(x,my,srcImage->width,bit,srcImage->bpp)]+
        algorithm[0][2]*srcImage->data[Index(px,my,srcImage->width,bit,srcImage->bpp)]+
        algorithm[1][0]*srcImage->data[Index(mx,y,srcImage->width,bit,srcImage->bpp)]+
        algorithm[1][1]*srcImage->data[Index(x,y,srcImage->width,bit,srcImage->bpp)]+
        algorithm[1][2]*srcImage->data[Index(px,y,srcImage->width,bit,srcImage->bpp)]+
        algorithm[2][0]*srcImage->data[Index(mx,py,srcImage->width,bit,srcImage->bpp)]+
        algorithm[2][1]*srcImage->data[Index(x,py,srcImage->width,bit,srcImage->bpp)]+
        algorithm[2][2]*srcImage->data[Index(px,py,srcImage->width,bit,srcImage->bpp)];

    return result;
}


void convolute(Image* srcImage,Image* destImage,Matrix algorithm){
    int num_threads = 4;
    pthread_t threads[num_threads];
    ThreadData data[num_threads];

    int rows_per_thread = srcImage->height / num_threads;

    for (int i = 0; i < num_threads; i++){
        data[i].srcImage = srcImage;
        data[i].destImage = destImage;
        data[i].algorithm = algorithm;
        data[i].start_row = i * rows_per_thread;
        data[i].end_row = (i == num_threads - 1) ? srcImage->height : (i+1)*rows_per_thread;

        pthread_create(&threads[i], NULL, thread_func, &data[i]);
    }

    for (int i = 0; i < num_threads; i++){
        pthread_join(threads[i], NULL);
    }
}

int Usage(){
    printf("Usage: image <filename> <type>\n");
    return -1;
}

enum KernelTypes GetKernelType(char* type){
    if (!strcmp(type,"edge")) return EDGE;
    else if (!strcmp(type,"sharpen")) return SHARPEN;
    else if (!strcmp(type,"blur")) return BLUR;
    else if (!strcmp(type,"gauss")) return GAUSE_BLUR;
    else if (!strcmp(type,"emboss")) return EMBOSS;
    else return IDENTITY;
}

int main(int argc,char** argv){
    long t1=time(NULL);

    if (argc!=3) return Usage();

    Image srcImage,destImage;
    srcImage.data=stbi_load(argv[1],&srcImage.width,&srcImage.height,&srcImage.bpp,0);
    if (!srcImage.data){
        printf("Error loading file\n");
        return -1;
    }

    destImage = srcImage;
    destImage.data=malloc(srcImage.width*srcImage.height*srcImage.bpp);

    convolute(&srcImage,&destImage,algorithms[GetKernelType(argv[2])]);

    stbi_write_png("output.png",destImage.width,destImage.height,destImage.bpp,destImage.data,destImage.width*destImage.bpp);

    stbi_image_free(srcImage.data);
    free(destImage.data);

    printf("Took %ld seconds\n",time(NULL)-t1);
    return 0;
}