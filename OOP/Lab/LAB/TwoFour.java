package LAB;

import java.util.PriorityQueue;

public class TwoFour {
    public static void main(String[] args) throws CloneNotSupportedException {
        MyPriorityQueue<Integer> q1 = new MyPriorityQueue<>();
        q1.offer(10);
        q1.offer(20);
        q1.offer(50);
        MyPriorityQueue<Integer> q2 = q1.clone();
        System.out.print("Queue 1: ");
        while (q1.size() > 0) {
            System.out.print(q1.remove() + " ");
        }
        System.out.println();

        System.out.print("Queue2: ");
        while (q2.size() > 0) {
            System.out.print(q2.remove() + " ");
        }
    }
}