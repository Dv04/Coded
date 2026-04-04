package utils;

import java.util.NoSuchElementException;

import utils.functions.Func1;

public class Or<A, B> {

	private final boolean isLeft;
	private final A left;
	private final B right;

	private Or(boolean isLeft, A left, B right) {
		this.isLeft = isLeft;
		this.left = left;
		this.right = right;
	}

	public static <A,B> Or<A,B> inl(A left) {
		return new Or<>(true, left, null);
	}
	
	public static <A,B> Or<A,B> inr(B right) {
		return new Or<>(false, null, right);
	}

	public <C> C map(Func1<A,C> f, Func1<B,C> g) {
		return isLeft ? f.apply(left) : g.apply(right);
	}

	public boolean isLeft() {
		return isLeft;
	}

	public boolean isRight() {
		return !isLeft;
	}

	public A getLeft() {
		if (isLeft) {
			return left;
		} else {
			throw new NoSuchElementException();
		}
	}

	public B getRight() {
		if (isLeft) {
			throw new NoSuchElementException();
		} else {
			return right;
		}
	}

	@Override
	public String toString() {
		if (isLeft) {
			return "Left(" + left.toString() + ")";
		} else {
			return "Right(" + right.toString() + ")";
		}
	}

}
