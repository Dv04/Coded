package LAB;
import java.util.PriorityQueue;
 
public class MYPriorityQueue<E> extends PriorityQueue<E> implements Cloneable, MYPriorityQueue<E> {
    @Override
    public MYPriorityQueue<E> clone() throws CloneNotSupportedException {
        MYPriorityQueue<E> temp = new MYPriorityQueue<>();
        temp.addAll((MYPriorityQueue<E>) super.clone());
        return temp;
    }
}
 