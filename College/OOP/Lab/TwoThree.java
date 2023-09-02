// Write a generic method that returns the minimum elements in a two dimensional array. 

import java.util.Scanner;

public class TwoThree {

    public static void main(String[] args) {
        try (Scanner input = new Scanner(System.in)) {
            System.out.println("Enter the row of the array: ");
            int row = input.nextInt();
            System.out.println("Enter the column of the array: ");
            int column = input.nextInt();
            Integer[][] array = new Integer[row][column];
            for (int i = 0; i < array.length; i++) {
                System.out.println("Enter the " + i + " row of the array: ");
                for (int j = 0; j < array[i].length; j++) {
                    array[i][j] = input.nextInt();
                }
            }
            System.out.println("The smallest integer in the array is " + smallest(array));
        }
    }

    public static <E extends Comparable<E>> E smallest(E[][] array) {

        E smallest = array[0][0];
        for (E[] array1 : array) {
            for (E array11 : array1) {
                if (array11.compareTo(smallest) < 0) {
                    smallest = array11;
                }
            }
        }

        return smallest;
    }

}
