// // (Que no = 3)
#include <stdio.h>

int main()
{
    float x, y;
    printf("Enter the mean water level : ");
    scanf("%f", &x);
    printf("Enter the water level of the tank of nuclear reactor : ");
    scanf("%f", &y);
    if (y < 0.4 * x)
    {
        printf("Water level in tank is within 40%% band of the mean water level.\n");
    }
    else
    {
        printf("Water level in tank is not within 40%% band of the mean water level.\n");
    }
    return 0;
}

// // (Que no = 16)
// #include <stdio.h>
// #include <string.h>
// #include <ctype.h>

// int main()
// {
//     char x[] = "Luca Modric", y[] = "EpizOotiOlogiEs";
//     int a, reminder;
//     for (int i = 0; i <= strlen(x); i++)
//     {
//         if (x[i] == '\0')
//         {
//             a = i;
//         }
//         else
//         {
//             continue;
//         }
//     }
//     for (int i = 0; i <= strlen(y); i++)
//     {
//         if (y[i] == 'a' || y[i] == 'e' || y[i] == 'i' || y[i] == 'o' || y[i] == 'u')
//         {
//             if (islower(y[i]))
//             {
//                 reminder = a % i;
//                 break;
//             }
//         }
//         else
//         {
//             continue;
//         }
//     }
//     printf("The reminder is :%d\n", reminder);
//     return 0;
// }

// // (Que no = 19)
// #include <stdio.h>

// int main()
// {
//     float p=32,v=2,m,t=77;
//     m= (p*v)/(0.37*(t+460));
//     printf("The mass of air is :%.4f pounds\n",m);
//     return 0;
// }

// // (Que no 20)
// #include <stdio.h>
// #include <math.h>

// int main()
// {
//     int x = 2, y = -1, z = 56, b = 10, c = 55;
//     float a = 0.75, result1, result2;

//     // (1)
//     result1 = pow(a, 2) - (4 * x) + (7 * z);
//     result2 = b - (7 * c) + (0.5 * pow(y, 2));
//     printf("The left expression gives:%.3f\n", result1);
//     printf("The right expression gives:%.3f\n", result2);
//     if (result1 > result2)
//     {
//         printf("The given expression is true.\n");
//     }
//     else
//     {
//         printf("The given expression is false.\n");
//     }

//     // (2)
//     result1 = sin(c) + cos(pow((pow(x, 2) + pow(y, 2)), 0.5));
//     result2 = 4*a;
//     printf("\nThe left expression gives:%.3f\n", result1);
//     printf("The right expression gives:%.3f\n", result2);
//     if (result1 <= result2)
//     {
//         printf("The given expression is true.\n");
//     }
//     else
//     {
//         printf("The given expression is false.\n");
//     }
//     return 0;
// }

// // (Que no = 21)
// #include <stdio.h>
// #include <string.h>
// #include <ctype.h>
// int factorial(char *ptr)
// {
//     int k=0;
//     double fac=1;
//     for (int i = 0; i < strlen(ptr); i++)
//     {
//         if (islower(ptr[i]))
//         {
//             k++;
//         }
//     }
//     for (int i = k; i >= 1; i--)
//     {
//         fac *= i;
//     }
//     return fac;
// }
// int main()
// {
//     char str[20]="VekariyaHeet";
//     // printf("Enter the string: ");
//     // scanf("%s" ,&str[20]);
//     // printf("%s\n",str);
//     printf("The factorial is : %ld\n",factorial(str));
//     return 0;
// }

// // (Que no = 22)
// #include <stdio.h>
// #include <math.h>

// int power(float velocity[],float area,float *p[])
// {
//     // int p[10];
//     for (int i = 0; i < 10; i++)
//     {

//         *p[i]=(0.5*area*pow(velocity[i],3));
//     // printf("%f \n",0.5*area*pow(velocity[i],3));
//         // printf("p=%.3f\n",*p[i]);
//     }
//     return *p[10];

// }
// int main()
// {
//     int diameter=120;
//     float area,p[10],velocity[]={12,10.5,11.4,4.5,6.32,7.31,8.10,10.01,2.41,4.5};

//     area = (M_PI*pow(diameter,2)*0.25);
//     printf("%f\n",area);
//     printf("The array of power is :");
//     power(velocity,area,&p);
//     for (int i = 0; i < 10; i++)
//     {
//         printf("%f\n",p[i]);
//     }

//     return 0;
// }

// // (Que no = 23)
// #include <stdio.h>
// int main()
// {
//     int a, b = 0;
//     static int c[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 0};
//     for (a = 0; a < 10; ++a)
//     {
//         if ((c[a] % 2) == 0)
//         {
//             b += c[a];
//         }
//     }
//     printf("%d", b);
// }

// // (Que no = 24)
// #include <stdio.h>
// #include <math.h>
// float geomean(int num)
// {
//     int x[num];
//     long double mul = 1;
//     float result;
//     for (int i = 0; i < num; i++)
//     {
//         x[i] = (2 * i) + 1;
//     }
//     for (int i = 0; i < num; i++)
//     {
//         mul *= x[i];
//     }
//     result = pow(mul, 1.0 / num);
//     return result;
// }

// int main()
// {
//     int num;
//     printf("Enter the number : ");
//     scanf("%d", &num);
//     printf("The geometric mean is : %.2f\n", geomean(num));
//     return 0;
// }