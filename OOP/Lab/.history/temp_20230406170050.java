// Write a test program that prompts the user to enter ten numbers, invoke a method to reverse the numbers, display the numbers

import java.util.Scanner;

public class temp {
    public static void main(String[] args) {
        Scanner input = new Scanner(System.in);
        int[] numbers = new int[10];
        System.out.print("Enter ten numbers: ");
        for (int i = 0; i < numbers.length; i++) {
            numbers[i] = input.nextInt();
        }
        reverse(numbers);
        System.out.println("The reversed array is: ");
        for (int i = numbers.length; i > 0; i++) {
            System.out.print(numbers[i] + " ");
        }
    }
    public static void reverse(int[] list) {
        int[] temp = new int[list.length];
        for (int i = 0, j = temp.length - 1; i < list.length; i++, j--) {
            temp[j] = list[i];
        }
        for (int i = 0; i < list.length; i++) {
            list[i] = temp[i];
        }
    }
}