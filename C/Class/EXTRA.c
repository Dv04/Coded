#include <stdio.h>
#include <math.h>
#include <ctype.h>

int main()
{
    int i,n,z,sum=0,sr,er;

    printf("\nEnter the starting range and ending range:  ");
    scanf("%d %d",&sr,&er);

    for(n=sr;n<er;n++)
    {
      z=n;

    while(n!=0)
    {
        sum=sum+pow(n%10,3);
        n=n/10;
        
    }

    if(z==sum)
    {
        printf("\nThe number %d is armstrong number",z);
    }
    
    } 

    
    return 0;
}