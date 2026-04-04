package utils.functions;

@FunctionalInterface
public interface Func2<A,B,R> {
    
    public R apply(A a, B b);

}
