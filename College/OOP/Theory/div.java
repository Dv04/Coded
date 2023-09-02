// Write a recursive method that returns the smallest integer in an array. Write a test program that prompts the user to enter an integer and display its product.

import java.util.Scanner;

public class div {
    public static void main(String[] args) {
        try (Scanner input = new Scanner(System.in)) {
            System.out.print("Enter a list of numbers: ");
            int[] list = new int[input.nextInt()];
            for (int i = 0; i < list.length; i++) {
                list[i] = input.nextInt();
            }
            System.out.println("The minimum number is " + min(list));
        }
    }
    public static int min(int[] list) {
        int min = list[0];
        for (int i = 1; i < list.length; i++) {
            if (list[i] < min) {
                min = list[i];
            }
        }
        return min;
    }
}