import java.util.Scanner;

public class temp {
    public static void main(String[] args) {
        try (Scanner input = new Scanner(System.in)) {
            System.out.println("Enter a letter: ");
            char letter = input.next().charAt(0);
            if (letter.equals("a") || letter.equals("e") || letter.equals("i") || letter.equals("o")
                    || letter.equals("u")) {
                System.out.println("The letter is a vowel");
            }
            else if (letter.equals("A") || letter.equals("E") || letter.equals("I") || letter.equals("O")
                    || letter.equals("U")) {
                System.out.println("The letter is a vowel");
            }
            else {
                System.out.println("The letter is a constant");
            }
        }
    }
}