// Write a recursive method that returns the smallest integer in an array. Write a test program that prompts the user to enter an integer and display its product.

import java.util.Scanner;

public class TwoTwo {

    public static void main(String[] args) {
        try (Scanner input = new Scanner(System.in)) {
            System.out.print("Enter an integer: ");
            int n = input.nextInt();
            System.out.println("The product of the digits is " + product(n));
        }
    }

    public static int product(int n) {
        if (n < 10) {
            return n;
        } else {
            return (n % 10) * product(n / 10);
        }
    }
    
}
