package dsl;

public interface Sink<A> {
	void next(A item);
	void end();
}
