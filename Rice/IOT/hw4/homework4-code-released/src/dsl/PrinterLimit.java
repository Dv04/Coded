package dsl;

public class PrinterLimit<A> implements Sink<A> {

	private final long limit;
	private long count = 0;

	public PrinterLimit(long limit) {
		if (limit < 0) {
			throw new IllegalArgumentException();
		}
		this.limit = limit;
	}

	@Override
	public void next(A item) {
		count += 1;
		if (count <= limit) {
			System.out.println("item " + count + ": " + item.toString());
		}
	}

	@Override
	public void end() {
		System.out.println("END");
	}

}
