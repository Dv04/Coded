package dsl;

import utils.functions.Func2;

// Aggregation (one output item when the stream ends).

public class Fold<A, B> implements Query<A, B> {

	// TODO

	public Fold(B init, Func2<B,A,B> op) {
		// TODO
	}

	@Override
	public void start(Sink<B> sink) {
		// TODO
	}

	@Override
	public void next(A item, Sink<B> sink) {
		// TODO
	}

	@Override
	public void end(Sink<B> sink) {
		// TODO
	}
	
}
