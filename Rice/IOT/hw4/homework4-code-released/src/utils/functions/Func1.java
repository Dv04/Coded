package utils.functions;

@FunctionalInterface
public interface Func1<A,R> {
    
    public R apply(A a);

}
