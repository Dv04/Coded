package dsl;

import java.util.function.Function;

// Apply a function elementwise.

public class Map<A,B> implements Query<A,B> {

	// TODO

	public Map(Function<A,B> op) {
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
