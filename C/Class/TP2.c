#include <stdio.h>
#include <math.h>
#include <string.h>
#include <ctype.h>

// float  h(char x[10],int d)
// {
//    int mul=1,i;
//    printf("\nEnter a string x: ");
//    scanf("%s",x);

//    for(i=0;i<strlen(x);i++)
//    {
//        if(islower(x[i]))
//        {
//            d++;
//        }
//    }

//    i=1;
//    while(i<=d)
//    {
//      mul*=i;
//      i++;
//    }
//    return mul;
// }

int main()
{
    // char x[10],i;
    // char y[]="HelloWor";
    //   printf("\nThe factorial is %.3f\n",h(x,i));

    float x = 2, y = -1, z = 56, a = 0.75, b = 10, c = 55;
    char i[10], j[10];
    // *i = (((a * a) - (4 * x) + (7 * z)) > (b - (7 * c) + (y * y / 2))) ? "True" : "False";
    memcpy(i, (((a * a) - (4 * x) + (7 * z)) > (b - (7 * c) + (y * y / 2))) ? "True" : "False", 10);
    printf("%s\n", i);

    // return 0;
}