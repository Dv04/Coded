import java.util.PriorityQueue;

public class MyPriorityQueue<E> extends PriorityQueue<E> implements Cloneable{
    @Override
    public MyPriorityQueue<E> clone() throws CloneNotSupportedException {
        MyPriorityQueue<E> temp = new MyPriorityQueue<>();
        @SuppressWarnings("unchecked")
        MyPriorityQueue<E> cloned = (MyPriorityQueue<E>) super.clone();
        temp.addAll(cloned);
        return temp;
    }
}
