import java.util.Scanner;

public class temp {
    public static void main(String[] args) {
        int factor = 2;
        Scanner input = new Scanner(System.in);
        System.out.print("Enter an integer: ");
        int number = input.nextInt();
        while (factor <= number) {
            if (number % factor == 0) {
                System.out.print(factor + " ");
                number /= factor;
            } else {
                factor++;
            }
        }
        System.out.println("\n");
    }
}