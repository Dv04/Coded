import java.util.Scanner;

public class TwoFour {
    public static void main(String[] args) throws CloneNotSupportedException {
        MyPriorityQueue<Integer> q1 = new MyPriorityQueue<>();
        try (Scanner sc = new Scanner(System.in)) {
            System.out.println("How many numbers do you want to input: ");
            int n = sc.nextInt();
            System.out.println("\n");
            while (n > 0) {
                System.out.println("Enter a number: ");
                int a = sc.nextInt();
                q1.offer(a);
                n--;
            }
        }
        MyPriorityQueue<Integer> q2 = q1.clone();
        System.out.println("\n\nQueue1: ");
        while (q1.size() > 0) {
            System.out.print(q1.poll() + " ");
        }
        System.out.println("\nQueue2: ");
        while (q2.size() > 0) {
            System.out.print(q2.poll() + " ");
        }

        System.out.println("\n");

    }
}
