package utils;

import java.util.NoSuchElementException;

import utils.functions.Func1;

public class Or3<A, B, C> {

	public final boolean isLeft;
	public final boolean isMid;
	public final boolean isRight;
	private final A left;
	private final B mid;
	private final C right;

	private Or3(boolean isL, boolean isM, boolean isR, A l, B m, C r) {
		this.isLeft = isL;
		this.isMid = isM;
		this.isRight = isR;
		this.left = l;
		this.mid = m;
		this.right = r;
	}

	public static <A,B,C> Or3<A,B,C> inl(A left) {
		return new Or3<>(true, false, false, left, null, null);
	}

	public static <A,B,C> Or3<A,B,C> inm(B mid) {
		return new Or3<>(false, true, false, null, mid, null);
	}
	
	public static <A,B,C> Or3<A,B,C> inr(C right) {
		return new Or3<>(false, false, true, null, null, right);
	}

	public <D> D map(Func1<A,D> f, Func1<B,D> g, Func1<C,D> h) {
		if (isLeft) {
			return f.apply(left);
		} else if (isMid) {
			return g.apply(mid);
		} else if (isRight) {
			return h.apply(right);
		} else {
			throw new NoSuchElementException();
		}
	}

	public A getLeft() {
		if (isLeft) {
			return left;
		} else {
			throw new NoSuchElementException();
		}
	}

	public B getMid() {
		if (isMid) {
			return mid;
		} else {
			throw new NoSuchElementException();
		}
	}

	public C getRight() {
		if (isRight) {
			return right;
		} else {
			throw new NoSuchElementException();
		}
	}

	@Override
	public String toString() {
		if (isLeft) {
			return "Left(" + left.toString() + ")";
		} else if (isMid) {
			return "Mid(" + mid.toString() + ")";
		} else if (isRight) {
			return "Right(" + right.toString() + ")";
		} else {
			throw new NoSuchElementException();
		}
	}

}
