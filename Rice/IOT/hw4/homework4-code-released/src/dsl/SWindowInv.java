package dsl;

import java.util.function.BinaryOperator;

// Efficient algorithm for aggregation over a sliding window.
// It assumes that there is a 'remove' operation for updating
// the aggregate when an element is evicted from the window.

public class SWindowInv<A> implements Query<A,A> {

	// TODO

	public SWindowInv
	(int wndSize, A init, BinaryOperator<A> insert, BinaryOperator<A> remove)
	{
		if (wndSize < 1) {
			throw new IllegalArgumentException("window size should be >= 1");
		}
		
		// TODO
	}

	@Override
	public void start(Sink<A> sink) {
		// TODO
	}

	@Override
	public void next(A item, Sink<A> sink) {
		// TODO
	}

	@Override
	public void end(Sink<A> sink) {
		// TODO
	}
	
}
