import java.util.Scanner;

public class temp {


    
    public static int gcd(int num1, int num2) {
        int gcd = 1;
        int k = 2;
        while (k <= num1 && k <= num2) {
            if (num1 % k == 0 && num2 % k == 0) {
                gcd = k;
            }
            k++;
        }
        return gcd;
    } 

}