/**
 * @file Print.c
 * @author Dev Sanghvi (dev04san@gmail.com)
 * @brief is it upper or lower
 * @version 1.0
 * @date 2022-04-27
 *
 * @copyright Copyright (c) 2022
 *
 */

#include <stdio.h>
#include <ctype.h>
#define x "Arsenal"

int main()
{
    // {
    // printf("Real Madrid is the best team in the world.\n");
    // char a[100];
    // printf("Enter a string: ");
    // scanf("%s", a);
    // if(isupper(a[0])){
    //     printf("Uppercase \n");
    // }
    // else{
    //     printf("Error \n");
    // }
    // return 0;
    // }

    // {
    // char x[20] = "BAYERN Munchen";
    // if (islower(x[0]) && islower(x[13]))
    // {
    //     printf("Lower\n");
    // }
    // return 0;
    // }


    for (size_t i = 0; i < 7; i++)
    {
        if (islower(x[i]))
        {
            printf(" %c\n", toupper(x[i]));
            // putchar(x[i]); // getchar/putchar is a function like printf designed to specifically handle character tokens in c.
        }
        else
        {
            continue;
        }
    }

    printf("\n");
    return 0;
}