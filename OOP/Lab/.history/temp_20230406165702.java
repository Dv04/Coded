import java.util.Scanner;

public class temp {

    public static void main(String[] args) {
        try (Scanner input = new Scanner(System.in)) {
            System.out.print("Enter Integer Value : ");
            int num1 = input.nextInt();
            System.out.print("Enter Integer Value : ");
            int num2 = input.nextInt();
            System.out.println("The GCD of " + num1 + " and " + num2 + " is " + gcd(num1, num2));
        }
    }

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