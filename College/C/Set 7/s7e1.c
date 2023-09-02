/**
 * @file s7e1.c
 * @author Dev Sanghvi (Dev04san@gmail.com)
 * @brief concatenate name, middle name, and surname into another string.
 * @version 1.0
 * @date 2022-06-16
 *
 * @copyright Copyright (c) 2022
 *
 */
#include <stdio.h>
#include <string.h>

int main()
{
    char nm[] = " Dev ", midnm[] = "Hardikkumar", surnm[] = " Sanghvi", result[100];
    strcat(result, surnm);
    strcat(result, nm);
    strcat(result, midnm);
    printf("The answer is : %s\n", result);
    return 0;
}
