package dsl;

public class Printer<A> implements Sink<A> {

	private long count = 0;

	@Override
	public void next(A item) {
		count += 1;
		System.out.println("item " + count + ": " + item.toString());
	}

	@Override
	public void end() {
		System.out.println("END");
	}

}
