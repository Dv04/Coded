package dsl;

// Duplicate each input item.

public class Dup<A> implements Query<A,A> {

	public Dup() {
		// nothing to do
	}

	@Override
	public void start(Sink<A> sink) {
		// nothing to do
	}

	@Override
	public void next(A item, Sink<A> sink) {
		sink.next(item);
		sink.next(item);
	}

	@Override
	public void end(Sink<A> sink) {
		sink.end();
	}
	
}
