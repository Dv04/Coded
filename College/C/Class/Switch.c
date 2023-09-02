/**
 * @file Untitled-1
 * @author Dev Sanghvi (dev04san@gmail.com)
 * @brief switch
 * @version 1.0
 * @date 2022-05-17
 *
 * @copyright Copyright (c) 2022
 *
 */

#include <stdio.h>
#include <ctype.h>
#include <string.h>

int main()
{
    // char k = '\0';
    // if (isupper(k))
    // {
    //     printf("%c\n", k);
    // }
    // switch (isupper(k))
    // {
    // case 1:
    //     printf("1%c\n", k);
    //     break;
    // case 0:
    //     printf("2%c\n", k);
    //     break;
    // default:
    //     printf("3%c\n", k);
    //     break;
    // }

    // int n[5];
    // for (int i = 0; i < 5; i++)
    // {
    //     printf("Enter %d number: ", i + 1);
    //     scanf("%d", &n[i]);
    //     if (n[i] < 0)
    //     {
    //         printf("\nYou entered a Negative Number\n\n");
    //         break;
    //     }
    // }

    char x[100];
    int count = 0;
    printf("Enter a string: ");
    scanf("%s", x);

    for (int i = 0; i < strlen(x); i++)
    {
        if (islower(x[i]))
        {
            count++;
        }
    }

    // if(count>0 && count%2==0 && strlen(x) % 3 == 0 && strlen(x) % 2 != 0) Wrong Code
    if (count>0 && count % 2 == 0 && (strlen(x) % 3 == 0) % 2 != 0)
    {
        printf("Errrrrorrrrr\n");
    }
    return 0;
}