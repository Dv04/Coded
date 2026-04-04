package dsl;

import java.text.DecimalFormat;

public class SLastCount<A> implements Sink<A> {

	private static DecimalFormat formatter = new DecimalFormat("#,###");

	public long count = 0;
	public A last = null;

	@Override
	public void next(A item) {
		count += 1;
		last = item;
	}

	@Override
	public void end() {
		System.out.println("# output items = " + formatter.format(count));
		System.out.println("last output item = " + last);
	}

}
