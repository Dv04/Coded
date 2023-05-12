// Write a method with following method header. public static int gcd(int num1, int num2) Write a program that prompts the user to enter two integers and compute the GCD of two integers.

import java.util.Scanner;

public class temp {

  public static void main(String[] args) {
    int div = 2;

    Scanner input = new Scanner(System.in);
    System.out.print("Enter Integer Value : ");
    int number = input.nextInt();
    
    while (number > 1) {
      if (number % div == 0) {
        System.out.print(div + ",");
        number = number / div;
      } else {
        div++;
      }
    }
  }

}