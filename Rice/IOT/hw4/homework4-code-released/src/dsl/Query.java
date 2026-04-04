package dsl;

public interface Query<A,B> {
	void start(Sink<B> sink);
	void next(A item, Sink<B> sink);
	void end(Sink<B> sink);
}
