/**
 * @file unstuffing.c
 * @author Dev Sanghvi (Dev04san@gmail.com)
 * @brief Bit unstuffing. If there is a 0 after 5 consecutive 1's remove the zero
 * @version 1.0
 * @date 2023-04-25
 *
 * @copyright Copyright (c) 2023
 *
 */

#include <stdio.h>
#include <string.h>
void bitDestuffing(int N, int arr[])
{
    int ans[30];
    int i, j, k;
    i = 0;
    j = 0;
    int count = 1;
    while (i < N)
    {
        if (arr[i] == 1)
        {
            ans[j] = arr[i];
            for (k = i + 1;
                 arr[k] == 1 && k < N && count < 5;
                 k++)
            {
                j++;
                ans[j] = arr[k];
                count++;
                if (count == 5)
                {
                    k++;
                }
                i = k;
            }
        }
        else
        {
            ans[j] = arr[i];
        }
        i++;
        j++;
    }
    printf("This is the output: ");
    for (i = 0; i < j; i++)
        printf("%d", ans[i]);
    printf("\n");
}
int main()
{
    int N = 7;
    int arr[] = {1, 1, 1, 1, 1,0, 1};
    printf("This is the input: ");
    for (int i = 0; i < N; i++)
        printf("%d", arr[i]);
    printf("\n");
    bitDestuffing(N, arr);
    return 0;
}