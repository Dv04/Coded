package dsl;

import utils.functions.Func2;

// Running aggregation (one output item per input item).

public class Scan<A, B> implements Query<A, B> {

	// TODO

	public Scan(B init, Func2<B,A,B> op) {
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
