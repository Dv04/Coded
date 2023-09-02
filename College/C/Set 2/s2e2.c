/**
 * @file s2e2.c
 * @author Dev Sanghvi (dev04san@gmail.com)
 * @brief Converting distance from km to m in feet.
 * @version 1.0
 * @date 2022-04-21
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include <stdio.h>

int main(){

    int km,m;
    float i,f;

    printf("Enter the distance between two cities in km: ");
    scanf("%d",&km);

    m = km * 1000;
    i = (float)m * 39.3701;
    f = (float)m * 3.28084;

    printf("Distance between two cities in m is %dm\n",m);
    printf("Distance between two cities in inches is %.3fin\n",i);
    printf("Distance between two cities in feet is %.3fft\n",f);

    return 0;
}