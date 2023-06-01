public class TwoFour {
    public static void main(string[] args) throws CloneNotSupportedException {
        MyPriorityQueue<Integer> q1 = new MyPriorityQueue<>();
        q1.offer(110);
        q1.offer(20);
        q1.offer(50);
        MyPriorityQueue<Integer> q2 = q1.clone();
    }
}
