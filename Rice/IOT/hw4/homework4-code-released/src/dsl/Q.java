package dsl;

import java.text.DecimalFormat;
import java.util.Iterator;
import java.util.NoSuchElementException;
import java.util.function.BinaryOperator;
import java.util.function.Function;
import java.util.function.Predicate;

import utils.Or;
import utils.functions.Func2;
import utils.functions.Func3;

public class Q {

	private Q() {

	}

	public static <A> Query<A,A> id() {
		return new Id<>();
	}

	public static <A> Query<A,A> dup() {
		return new Dup<>();
	}

	public static <A> Query<A,A> ignore(int n) {
		return new Ignore<>(n);
	}
	
	public static <A,B> Query<A,B> map(Function<A,B> op) {
		return new Map<>(op);
	}

	public static <A> Query<A,A> emit(int n, A value) {
		return new Emit<>(n, value);
	}

	public static <A> Query<A,A> filter(Predicate<A> pred) {
		return new Filter<>(pred);
	}

	public static <A,B> Query<A,B> fold(B init, Func2<B,A,B> op) {
		return new Fold<>(init, op);
	}

	private static class SumCount {
		public static final SumCount zero = new SumCount(0.0, 0);
		public final double s;
		public final long c;
		public SumCount(double s, long c) {
			this.s = s;
			this.c = c;
		}
		public SumCount add(SumCount p) {
			return new SumCount(s + p.s, c + p.c);
		}
	}

	public static Query<Double,Double> foldAvg() {
		Query<Double,SumCount> q1 = Q.map(x -> new SumCount(x, 1));
		Query<SumCount,SumCount> q2 = Q.fold(SumCount.zero, SumCount::add);
		Query<SumCount,Double> q3 = Q.map(p -> p.s / p.c);
		return Q.pipeline(q1, q2, q3);
	}

	private static class SumSqCount {
		public static final SumSqCount zero = new SumSqCount(0.0, 0.0, 0);
		public final double s2;
		public final double s;
		public final long c;
		public SumSqCount(double s2, double s, long c) {
			this.s2 = s2;
			this.s = s;
			this.c = c;
		}
		public SumSqCount add(SumSqCount p) {
			return new SumSqCount(s2 + p.s2, s + p.s, c + p.c);
		}
	}

	public static Query<Double,Double> foldStdev() {
		Query<Double,SumSqCount> q1 =
			Q.map(x -> new SumSqCount(x*x, x, 1));
		Query<SumSqCount,SumSqCount> q2 =
			Q.fold(SumSqCount.zero, SumSqCount::add);
		Query<SumSqCount,Double> q3 =
			Q.map(p -> {
				double mu = p.s / p.c;
				return Math.sqrt((p.s2 / p.c) - mu*mu);
			});
		return Q.pipeline(q1, q2, q3);
	}

	public static <A,B> Query<A,B> scan(B init, Func2<B,A,B> op) {
		return new Scan<>(init, op);
	}

	public static <A,B> Query<A,B>
	sWindowNaive(int wndSize, B init, Func2<B,A,B> op)
	{
		return new SWindowNv<>(wndSize, init, op);
	}

	public static <A> Query<A,A>
	sWindowInv(int wndSize, A init, BinaryOperator<A> insert,
			   BinaryOperator<A> remove)
	{
		return new SWindowInv<>(wndSize, init, insert, remove);
	}

	public static <A,B> Query<A,B> sWindow2(Func2<A,A,B> op) {
		return new SWindow2<>(op);
	}

	public static <A,B> Query<A,B> sWindow3(Func3<A,A,A,B> op) {
		return new SWindow3<>(op);
	}

	public static <A,B,C,D> Query<A,D>
	parallel(Query<A,B> q1, Query<A,C> q2, Func2<B,C,D> op)
	{
		return new Parallel<>(q1, q2, op);
	}

	public static <A,B,C> Query<A,C>
	pipeline(Query<A,B> q1, Query<B,C> q2)
	{
		return new Pipeline<>(q1, q2);
	}

	public static <A,B,C,D> Query<A,D>
	pipeline(Query<A,B> q1, Query<B,C> q2, Query<C,D> q3)
	{
		return pipeline(q1, pipeline(q2, q3));
	}

	public static <A,B,C,D,E> Query<A,E>
	pipeline(Query<A,B> q1, Query<B,C> q2, Query<C,D> q3, Query<D,E> q4)
	{
		return pipeline(q1, pipeline(q2, q3, q4));
	}

	public static <A,B,C,D,E,F> Query<A,F>
	pipeline(Query<A,B> q1, Query<B,C> q2, Query<C,D> q3,
			 Query<D,E> q4, Query<E,F> q5)
	{
		return pipeline(q1, pipeline(q2, q3, q4, q5));
	}

	public static <A,B,C,D,E,F,G> Query<A,G>
	pipeline(Query<A,B> q1, Query<B,C> q2, Query<C,D> q3,
			 Query<D,E> q4, Query<E,F> q5, Query<F,G> q6)
	{
		return pipeline(q1, pipeline(q2, q3, q4, q5, q6));
	}

	public static <A,B> Query<A,B> loop(Query<Or<A,B>,B> q) {
		return new Loop<>(q);
	}

	public static Iterator<Integer> intStream(int n) {
		if (n < 0) {
			throw new IllegalArgumentException("n must be >= 0");
		}
		return new Iterator<Integer>() {
			private int index = 1;
			@Override
			public boolean hasNext() {
				return index <= n;
			}
			@Override
			public Integer next() {
				if (index > n) {
					throw new NoSuchElementException();
				}
				int out = index;
				index += 1;
				return out;
			}
		};
	}

	public static <A,B> long execute(Iterator<A> it, Query<A,B> q, Sink<B> sink)
	{
		long n = 0;
		long start = System.nanoTime();

		q.start(sink);
		while (it.hasNext()) {
			A item = it.next();
			q.next(item, sink);
			n += 1;
		}
		q.end(sink);
		
		long end = System.nanoTime();
		
		DecimalFormat formatter = new DecimalFormat("#,###");
		long timeNano = end - start;
		long timeMsec = timeNano / 1_000_000;
		System.out.println("duration = " + formatter.format(timeMsec) + " msec");
		long throughput = (n * 1000L * 1000 * 1000) / timeNano;
		System.out.println("throughput = " + formatter.format(throughput) + " tuples/sec");

		return throughput;
	}

}
