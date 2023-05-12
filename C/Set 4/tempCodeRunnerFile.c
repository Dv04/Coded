
#include <stdio.h>

int main()
{
    int num;

    printf("How many Students are there: ");
    scanf("%d", &num);

    int Roll[num], Mar[num][5], Tot[num];
    float Per[num];
    char Res[2][5] = {"Pass", "Fail"}, Sub[5][10] = {"Maths", "EGD", "ENG", "BE", "PPS"};
    for (int j = 0; j < num; j++)
    {
        printf("\nEnter Roll Number: ");
        scanf("%d", &Roll[j]);

        for (int i = 0; i < 5; i++)
        {
            printf("Enter marks of subject %s: ", Sub[i]);
            scanf("%d", &Mar[j][i]);
        }

        Tot[j] = Mar[j][0] + Mar[j][1] + Mar[j][2] + Mar[j][3] + Mar[j][4];
        Per[j] = (float)Tot[j] / 5;
    }
    for (int j = 0; j < num; j++)
    {
        printf("\nHello Student %d\nYour Roll No is: %d\n", j + 1, Roll[j]);
        for (int i = 0; i < 5; i++)
        {
            printf("Marks for subject %s: %d\n", Sub[i], Mar[j][i]);
        }
        printf("Your scored:\n\t%d Marks in Total\n\t%.2f%% Percentage\n", Tot[j], Per[j]);
        if (Per[j] < 33)
        {
            printf("You have Failed\n\n");
        }
        else
        {
            printf("You have Passed\n\n");
        }
    }
    return 0;
}