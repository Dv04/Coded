package dsl;

// The identity transformation.

public class Id<A> implements Query<A,A> {

	public Id() {
		// nothing to do
	}

	@Override
	public void start(Sink<A> sink) {
		// nothing to do
	}

	@Override
	public void next(A item, Sink<A> sink) {
		sink.next(item);
	}

	@Override
	public void end(Sink<A> sink) {
		sink.end();
	}
	
}
