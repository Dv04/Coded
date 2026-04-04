package dsl;

public class S {

	private S() {

	}

	public static <A> Printer<A> printer() {
		return new Printer<>();
	}

	public static <A> PrinterLimit<A> printerLimit(long limit) {
		return new PrinterLimit<>(limit);
	}

	public static <A> SLastCount<A> lastCount() {
		return new SLastCount<>();
	}

}
