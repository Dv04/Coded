package dsl;

// Emit a value in the beginning (n times) and then echo
// the input stream.

public class Emit<A> implements Query<A,A> {

	// TODO

	public Emit(int n, A value) {
		if (n < 0) {
			throw new IllegalArgumentException("Emit: n must be >= 0");
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
