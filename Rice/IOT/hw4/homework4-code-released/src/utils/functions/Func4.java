package utils.functions;

@FunctionalInterface
public interface Func4<A,B,C,D,R> {
    
    public R apply(A a, B b, C c, D d);

}
