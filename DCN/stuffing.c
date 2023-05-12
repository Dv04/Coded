#include <stdio.h>
#include <string.h>

void bitStuffing(int N, int arr[])
{
    int ans[30];
    int i, j, k;
    i = 0;
    j = 0;
    while (i < N)
    {
        if (arr[i] == 1)
        {
            int count = 1;
            ans[j] = arr[i];
            for (k = i + 1;
                 arr[k] == 1 && k < N && count < 5; k++)
            {
                j++;
                ans[j] = arr[k];
                count++;
                if (count == 5)
                {
                    j++;
                    ans[j] = 0;
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
    int N = 6;
    int arr[] = {1, 1, 1, 1, 1, 1};
    printf("This is the input: ");
    for (int i = 0; i < N; i++)
        printf("%d", arr[i]);
    printf("\n");
    bitStuffing(N, arr);
    return 0;
}