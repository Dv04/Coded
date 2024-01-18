// Take two arrays. char a[26] = "abcdefghijklmnopqrstuvwxyz" and "pqlajskdiwuemrntyvgchbxz"

// take a string and encrypt it with the arrays, by replacing the letters with corresponding characters from second aray.

#include <stdio.h>
#include <string.h>

int main()
{
    char a[37] = "ab1cde2fgh4ijk5 lm3nop6qr8st9uv7wx0yz";
    char b[37] = "pqow0lak9sieu8rj7dhf6yg5m4 t3nbv2cx1z";
    char str[100] = "hey there how are you Number is 9879200470";
    int k = 10;
    int i, j;
    for (i = 0; i < strlen(str); i++)
    {
        for (j = 0; j < 37; j++)
        {
            if (str[i] == a[j])
            {
                printf("%d-%d ", j, j + k);
                str[i] = b[(j + k) % 37];
                break;
            }
        }
    }
    printf("\nThe encrypted string is: %s", str);

    for (int i = 0; i < strlen(str); i++)
    {
        for (int j = 0; j < 37; j++)
        {
            if (str[i] == b[j])
            {
                str[i] = a[(j - k + 37) % 37];
                break;
            }
        }
    }

    printf("\nThe decrypted string is: %s\n", str);

    return 0;
}