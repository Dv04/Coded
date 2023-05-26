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
            System.out.println("The smallest integer in the array is " + smallestInteger(array, 0));
        }

    }

    public static int smallestInteger(int[] array, int startIndex) {
        // base case: the size of the array equals the start index
        if (startIndex == array.length) {
            return 100000;
        }
        // get the smallest integer from the remaining elements
        int smallest = smallestInteger(array, startIndex + 1);
        // compare it to the current element and return the smaller one
        if (array[startIndex] < smallest) {
            return array[startIndex];
        } else {
            return smallest;
        }
    }

}