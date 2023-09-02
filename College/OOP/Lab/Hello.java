
import java.io.PrintStream;

public class Hello {
    public static void main(String[] args) {
        extracted().println("Welcome to Java, Learning Java Now and Programming is fun.");
    }

    private static PrintStream extracted() {
        return System.out;
    }
}