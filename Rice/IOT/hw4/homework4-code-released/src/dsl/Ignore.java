package dsl;

// Consume and discard the first n elements and then echo
// the input stream.

public class Ignore<A> implements Query<A,A> {

	private final int n;
	private int i;

	public Ignore(int n) {
		if (n < 0) {
			throw new IllegalArgumentException("Emit: n must be >= 0");
		}
		this.n = n;
		this.i = 0;
	}

	@Override
	public void start(Sink<A> sink) {
		i = 0;
	}

	@Override
	public void next(A item, Sink<A> sink) {
		if (i < n) {
			i++;
		} else {
			sink.next(item);
		}
	}

	@Override
	public void end(Sink<A> sink) {
		sink.end();
	}
	
}
