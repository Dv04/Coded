import java.util.Scanner;

// Write a program that prompts the user to enter a letter and check whether a letter is a vowel or constant.

public class temp {
    public static void main(String[] args) {
        try (Scanner input = new Scanner(System.in)) {
            System.out.println("Enter a letter: ");
            String letter = input.nextLine();
            if (letter.equals("a") || letter.equals("e") || letter.equals("i") || letter.equals("o")
                    || letter.equals("u")) {
                System.out.println("The letter is a vowel");
            }
            
            else {
                System.out.println("The letter is a constant");
            }
        }
    }
}