/**
 * @file DyMeAl.c
 * @author Dev Sanghvi (Dev04san@gmail.com)
 * @brief Dynamic Memory Allocation
 * @version 1.0
 * @date 2022-06-28
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include <stdio.h>
#include <stdlib.h>
int main(){

    int a = 1;
    
    int *b = (int*)malloc(100*sizeof(int)); 
    printf("a = %lu\n", sizeof(b));
    return 0;
}



/*  We use dynamic memory allocation via use of a pointer variable to represent a data structure with some initial memory assignments before the data structure element is processed.
    Calloc() takes a little linger than malloc() because of the extra step of initializing the allocated memory by 0 but in practise the difference in speed is very tiny and thus not recognizable.
    Malloc takes a single argument while calloc() takes two
    Malloc does not initialize while calloc intializes at 0.
    Realloc is used for modifying size for previously allocated memory space. 
    Free is used to relief or deallocate memory in c to make space for indefined variable or you can make the us of space more and /or make program faster.

    A null pointer is a pointer which points to nothing and i sused in dynaic memory allocation.
*/