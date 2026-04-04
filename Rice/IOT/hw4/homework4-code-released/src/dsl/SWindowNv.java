package dsl;

import utils.functions.Func2;

// Naive algorithm for aggregation over a sliding window.

public class SWindowNv<A,B> implements Query<A,B> {

	// TODO

	public SWindowNv(int wndSize, B init, Func2<B,A,B> op) {
		if (wndSize < 1) {
			throw new IllegalArgumentException("window size should be >= 1");
		}

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
