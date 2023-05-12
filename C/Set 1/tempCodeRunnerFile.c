#include <stdio.h>
#include <math.h>
#define PI 3.14
int main()
{
    float r, peri, area;
    r = sqrt((pow(4, 2) + (pow(5, 2))));
    peri = 2 * PI * r;
    area = r * r * PI;
    printf("Perimeter = %.4f\nArea = %.4f\n", peri, area);
}
