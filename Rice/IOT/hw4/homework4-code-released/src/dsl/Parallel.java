package dsl;

import utils.functions.Func2;

// A variant of parallel composition, which is similar to 'zip'.

public class Parallel<A, B, C, D> implements Query<A, D> {

	// TODO

	public Parallel(Query<A,B> q1, Query<A,C> q2, Func2<B,C,D> op) {
		// TODO
	}

	@Override
	public void start(Sink<D> sink) {
		// TODO
	}

	@Override
	public void next(A item, Sink<D> sink) {
		// TODO
	}

	@Override
	public void end(Sink<D> sink) {
		// TODO
	}
	
}
