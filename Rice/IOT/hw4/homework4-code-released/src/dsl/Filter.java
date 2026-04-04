package dsl;

import java.util.function.Predicate;

// Filter out elements that falsify the given predicate.

public class Filter<A> implements Query<A,A> {

	// TODO

	public Filter(Predicate<A> pred) {
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
