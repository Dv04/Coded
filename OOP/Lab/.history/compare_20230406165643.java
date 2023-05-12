import java.util.Scanner;

public class compare {
        public static void main(String[] args) {
        try (Scanner input = new Scanner(System.in)) {
            System.out.print("Enter first integer: ");
            int X = input.nextInt();
            
            System.out.print("Enter second integer: ");
            int Y = input.nextInt();
            
            int gcd = gcd(X, Y);
            System.out.println("GCD of " + X + " and " + Y + " is " + gcd);
        }
    }
    public static int gcd(int X, int Y) {
        
        if (X < Y) {
        int temp = X;
        X = Y;
        Y = temp;
        }
        
        while (Y != 0) {
        int remainder = X % Y;
        X = Y;
        Y = remainder;
        }
        return X;
    }
}
