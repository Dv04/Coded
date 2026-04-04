package dsl;

import utils.Or;

// Feedback composition.

public class Loop<A,B> implements Query<A,B> {

	// TODO

	public Loop(Query<Or<A,B>,B> q) {
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
