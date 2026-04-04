package dsl;

// Serial composition.

public class Pipeline<A,B,C> implements Query<A,C> {

	// TODO

	public Pipeline(Query<A,B> q1, Query<B,C> q2) {
		// TODO
	}

	@Override
	public void start(Sink<C> sink) {
		// TODO
	}

	@Override
	public void next(A item, Sink<C> sink) {
		// TODO
	}

	@Override
	public void end(Sink<C> sink) {
		// TODO
	}
	
}
