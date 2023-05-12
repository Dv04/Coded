// Write a recursive method that returns the smallest integer in an array.

import java.util.Scanner;

public class TwoTwoTwo {

    public static void main(String[] args) {
        try (Scanner input = new Scanner(System.in)) {
            System.out.print("Enter the array size: ");
            int[] array = new int[input.nextInt()];
            for (int i = 0; i < array.length; i++) {
                System.out.print("Enter an integer: ");
                array[i] = input.nextInt();
            }
            System.out.println("The smallest integer in the array is " + smallest(array));
        }

    }

    public static int smallest(int[] array) {

        int smallest = array[0];
        for (int i = 1; i < array.length; i++) {
            if (array[i] < smallest) {
                smallest = array[i];
            }
        }

        return smallest;
    }

}