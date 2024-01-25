// Take two arrays. char a[26] = "abcdefghijklmnopqrstuvwxyz" and "pqlajskdiwuemrntyvgchbxz"
// take a string and encrypt it with the arrays, by replacing the letters with corresponding characters from second aray.

#include <stdio.h>
#include <string.h>

void encrypt(char *str)
{
    for (int i = 0; i < strlen(str); i++)
    {
        str[i] = str[i] + i;
    }
}

void decrypt(char *str)
{
    for (int i = 0; i < strlen(str); i++)
    {
        str[i] = str[i] - i;
    }
}

int main()
{
    char str[100] = "heyyyy there how are you Number is 9879200470";
    encrypt(str);
    printf("\nThe encrypted string is: %s", str);
    decrypt(str);
    printf("\nThe decrypted string is: %s\n", str);
    return 0;
}

// for (i = 0; i < strlen(str); i++)
//     {
//         for (j = 0; j < 37; j++)
//         {
//             if (str[i] == a[j])
//             {
//                 printf("%d-%d ", j, j + k);
//                 str[i] = b[(j + k) % 37];
//                 break;
//             }
//         }
//     }
//     printf("\nThe encrypted string is: %s", str);

//     for (int i = 0; i < strlen(str); i++)
//     {
//         for (int j = 0; j < 37; j++)
//         {
//             if (str[i] == b[j])
//             {
//                 str[i] = a[(j - k + 37) % 37];
//                 break;
//             }
//         }
//     }

//     printf("\nThe decrypted string is: %s\n", str);