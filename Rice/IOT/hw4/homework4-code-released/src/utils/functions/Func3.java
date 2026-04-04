package utils.functions;

@FunctionalInterface
public interface Func3<A,B,C,R> {
    
    public R apply(A a, B b, C c);

}
