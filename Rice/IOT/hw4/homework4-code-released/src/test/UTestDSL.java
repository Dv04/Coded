package test;

import static org.junit.Assert.*;

import java.util.Iterator;

import org.junit.Before;
import org.junit.Test;

import dsl.*;
import utils.Or;

public class UTestDSL {

	@Before
	public void setUp() {
		// nothing to do
	}

	@Test
	public void testDup() {
		System.out.println("***** Test Dup *****");
	
		Query<Integer,Integer> q = Q.dup();
		SLastCount<Integer> sink = S.lastCount();
		
		int n = 1000;
		q.start(sink);
		assertEquals(0, sink.count);
		for (int i=1; i<=n; i++) {
			q.next(i, sink);
			assertEquals(2L*i, sink.count);
			assertEquals(i, sink.last.intValue());
		}
		q.end(sink);
		assertEquals(2*n, sink.count);
		assertEquals(n, sink.last.intValue());
	}

	@Test
	public void testEmit() {
		System.out.println("***** Test Emit *****");
	
		int k = 5;
		int v = 100;
		Query<Integer,Integer> q = Q.emit(k, v);
		SLastCount<Integer> sink = S.lastCount();
		
		int n = 1000;
		q.start(sink);
		assertEquals(k, sink.count);
		assertEquals(v, sink.last.intValue());
		for (int i=1; i<=n; i++) {
			q.next(i, sink);
			assertEquals(k + i, sink.count);
			assertEquals(i, sink.last.intValue());
		}
		q.end(sink);
		assertEquals(k + n, sink.count);
		assertEquals(n, sink.last.intValue());
	}

	@Test
	public void testFilter() {
		System.out.println("***** Test Filter *****");
	
		Query<Integer,Integer> q = Q.filter(x -> x % 2 == 0);
		SLastCount<Integer> sink = S.lastCount();
		
		int n = 1000;
		q.start(sink);
		assertEquals(0, sink.count);
		for (int i=1; i<=n; i++) {
			q.next(i, sink);
			assertEquals(i/2, sink.count);
			if (i >= 2) {
				assertEquals(2*(i/2), sink.last.intValue());
			}
		}
		q.end(sink);
		assertEquals(n/2, sink.count);
		assertEquals(n, sink.last.intValue());
	}

	@Test
	public void testFold() {
		System.out.println("***** Test Fold *****");
	
		Query<Integer,Integer> q = Q.fold(0, (x, y) -> x + y);
		SLastCount<Integer> sink = S.lastCount();
		
		int n = 1000;
		q.start(sink);
		assertEquals(0, sink.count);
		for (int i=1; i<=n; i++) {
			q.next(i, sink);
			assertEquals(0, sink.count);
		}
		q.end(sink);
		assertEquals(1, sink.count);
		assertEquals(((1+n)*n)/2, sink.last.intValue());
	}

	@Test
	public void testId() {
		System.out.println("***** Test Id *****");
	
		Query<Integer,Integer> q = Q.id();
		SLastCount<Integer> sink = S.lastCount();
		
		int n = 1000;
		q.start(sink);
		assertEquals(0, sink.count);
		for (int i=1; i<=n; i++) {
			q.next(i, sink);
			assertEquals(i, sink.count);
			assertEquals(i, sink.last.intValue());
		}
		q.end(sink);
		assertEquals(n, sink.count);
		assertEquals(n, sink.last.intValue());
	}

	@Test
	public void testIgnore() {
		System.out.println("***** Test Ignore *****");
	
		int k = 50;
		Query<Integer,Integer> q = Q.ignore(k);
		SLastCount<Integer> sink = S.lastCount();
		
		int n = 1000;
		q.start(sink);
		for (int i=1; i<=n; i++) {
			q.next(i, sink);
			assertEquals(Integer.max(i - k, 0), sink.count);
			if (i > k) {
				assertEquals(i, sink.last.intValue());
			}
		}
		q.end(sink);
		assertEquals(Integer.max(n - k, 0), sink.count);
		if (n > k) {
			assertEquals(n, sink.last.intValue());
		}
	}

	@Test
	public void testLoop1() {
		System.out.println("***** Test Loop (1) *****");
	
		Query<Or<Long,Long>,Or<Long,Long>> q1 = Q.filter(Or::isRight);
		Query<Or<Long,Long>,Long> q2 = Q.map(Or::getRight);
		Query<Long,Long> q = Q.loop(Q.pipeline(q1, q2));
		SLastCount<Long> sink = S.lastCount();
		
		int n = 1000;
		q.start(sink);
		assertEquals(0, sink.count);
		for (long i=1; i<=n; i++) {
			q.next(i, sink);
			assertEquals(0, sink.count);
		}
		q.end(sink);
		assertEquals(0, sink.count);
	}

	@Test
	public void testLoop2() {
		System.out.println("***** Test Loop (2) *****");
	
		Query<Or<Long,Long>,Long> q1 = Q.map(x -> x.map(y -> y, y -> y));
		Query<Long,Long> q = Q.loop(q1);
		SLastCount<Long> sink = S.lastCount();
		
		q.start(sink);
		assertEquals(0, sink.count);
		// If you uncomment the next line then the program will diverge.
		// q.next(1L, sink);
	}

	@Test
	public void testLoop3() {
		System.out.println("***** Test Loop (3) *****");
	
		Query<Or<Long,Long>,Long> q1 =
			Q.pipeline(Q.filter(Or::isLeft), Q.map(Or::getLeft));
		Query<Or<Long,Long>,Long> q2 =
			Q.pipeline(Q.filter(Or::isRight), Q.map(Or::getRight));
		Query<Or<Long,Long>,Long> q3 =
			Q.parallel(q1, q2, (x, y) -> x + y);
		Query<Or<Long,Long>,Long> q4 =
			Q.pipeline(q3, Q.emit(1, 0L));
		Query<Long,Long> q = Q.loop(q4);
		SLastCount<Long> sink = S.lastCount();
		
		int n = 1000;
		q.start(sink);
		assertEquals(1L, sink.count);
		assertEquals(0L, sink.last.longValue());
		for (long i=1; i<=n; i++) {
			q.next(i, sink);
			assertEquals(i + 1, sink.count);
			assertEquals(((1+i)*i)/2, sink.last.longValue());
		}
		q.end(sink);
		assertEquals(n + 1L, sink.count);
		assertEquals(((1+n)*n)/2, sink.last.longValue());
	}

	@Test
	public void testLoop4() {
		System.out.println("***** Test Loop (4) *****");
	
		Query<Or<Long,Long>,Long> q1 =
			Q.pipeline(Q.filter(Or::isRight), Q.map(Or::getRight));
		Query<Or<Long,Long>,Long> q2 =
			Q.pipeline(q1, Q.dup(), Q.emit(1, 8L), Q.emit(1, 7L));
		Query<Long,Long> q = Q.loop(q2);

		long limit = 100;
		Sink<Long> sink = S.printerLimit(limit);
		
		// If you uncomment the next line then the program will never terminate.
		// It will print the first 'limit' elements of the infinite sequence
		// 7 8 7 7 8 8 7 7 7 7 8 8 8 8 ...
		//q.start(sink);
	}

	@Test
	public void testMap() {
		System.out.println("***** Test Map *****");
	
		Query<Integer,Integer> q = Q.map(x -> 2*x);
		SLastCount<Integer> sink = S.lastCount();
		
		int n = 1000;
		q.start(sink);
		assertEquals(0, sink.count);
		for (int i=1; i<=n; i++) {
			q.next(i, sink);
			assertEquals(i, sink.count);
			assertEquals(2*i, sink.last.intValue());
		}
		q.end(sink);
		assertEquals(n, sink.count);
		assertEquals(2*n, sink.last.intValue());
	}

	@Test
	public void testParallel1() {
		System.out.println("***** Test Parallel (1) *****");
	
		Query<Integer,Integer> q1 = Q.map(x -> 2*x);
		Query<Integer,Integer> q2 = Q.map(x -> 3*x);
		Query<Integer,Integer> q = Q.parallel(q1, q2, (x,y) -> x + y);
		SLastCount<Integer> sink = S.lastCount();
		
		int n = 1000;
		q.start(sink);
		assertEquals(0, sink.count);
		for (int i=1; i<=n; i++) {
			q.next(i, sink);
			assertEquals(i, sink.count);
			assertEquals(5*i, sink.last.intValue());
		}
		q.end(sink);
		assertEquals(n, sink.count);
		assertEquals(5*n, sink.last.intValue());
	}

	@Test
	public void testParallel2() {
		System.out.println("***** Test Parallel (2) *****");
	
		Query<Integer,Integer> q1 = Q.filter(x -> x % 2 == 1);
		Query<Integer,Integer> q2 = Q.filter(x -> x % 2 == 0);
		Query<Integer,Integer> q = Q.parallel(q1, q2, (x,y) -> x + y);
		SLastCount<Integer> sink = S.lastCount();
		
		int n = 1000;
		q.start(sink);
		assertEquals(0, sink.count);
		for (int i=1; i<=n; i++) {
			q.next(i, sink);
			assertEquals(i/2, sink.count);
			if (i % 2 == 0) {
				assertEquals(2*i-1, sink.last.intValue());
			}
		}
		q.end(sink);
		assertEquals(n/2, sink.count);
		assertEquals(2*n-1, sink.last.intValue());
	}

	@Test
	public void testParallel3() {
		System.out.println("***** Test Parallel (3) *****");
	
		Query<Integer,Integer> q1 = Q.filter(x -> x % 2 == 0);
		Query<Integer,Integer> q = Q.parallel(q1, Q.id(), (x,y) -> x + y);
		SLastCount<Integer> sink = S.lastCount();
		
		int n = 1000;
		q.start(sink);
		assertEquals(0, sink.count);
		for (int i=1; i<=n; i++) {
			q.next(i, sink);
			assertEquals(i/2, sink.count);
			if (i % 2 == 0) {
				assertEquals(3*i/2, sink.last.intValue());
			}
		}
		q.end(sink);
		assertEquals(n/2, sink.count);
		assertEquals(3*n/2, sink.last.intValue());
	}

	@Test
	public void testPipeline() {
		System.out.println("***** Test Pipeline *****");
	
		Query<Integer,Integer> q1 = Q.map(x -> 2*x);
		Query<Integer,Integer> q2 = Q.map(x -> 2*x);
		Query<Integer,Integer> q3 = Q.map(x -> 2*x);
		Query<Integer,Integer> q = Q.pipeline(q1, q2, q3);
		SLastCount<Integer> sink = S.lastCount();
		
		int n = 1000;
		q.start(sink);
		assertEquals(0, sink.count);
		for (int i=1; i<=n; i++) {
			q.next(i, sink);
			assertEquals(i, sink.count);
			assertEquals(8*i, sink.last.intValue());
		}
		q.end(sink);
		assertEquals(n, sink.count);
		assertEquals(8*n, sink.last.intValue());
	}

	@Test
	public void testScan() {
		System.out.println("***** Test Scan *****");
	
		Query<Integer,Integer> q = Q.scan(0, (x, y) -> x + y);
		SLastCount<Integer> sink = S.lastCount();
		
		int n = 1000;
		q.start(sink);
		assertEquals(0, sink.count);
		for (int i=1; i<=n; i++) {
			q.next(i, sink);
			assertEquals(i, sink.count);
			assertEquals((1+i)*i/2, sink.last.intValue());
		}
		q.end(sink);
		assertEquals(n, sink.count);
	}

	@Test
	public void testWindow2() {
		System.out.println("***** Test Window 2 *****");
	
		Query<Integer,Integer> q1 = Q.sWindowNaive(2, 0, Integer::sum);
		SLastCount<Integer> sink1 = S.lastCount();

		Query<Integer,Integer> q2 = Q.sWindow2(Integer::sum);
		SLastCount<Integer> sink2 = S.lastCount();
		
		int n = 1000;
		Iterator<Integer> it = Q.intStream(n);
		q1.start(sink1);
		q2.start(sink2);
		while (it.hasNext()) {
			Integer item = it.next();
			q1.next(item, sink1);
			q2.next(item, sink2);
			assertEquals(sink1.count, sink2.count);
			assertEquals(sink1.last, sink2.last);
		}
		q1.end(sink1);
		q2.end(sink2);
		assertEquals(sink1.count, sink2.count);
		assertEquals(sink1.last, sink2.last);
	}

	@Test
	public void testWindow3() {
		System.out.println("***** Test Window 3 *****");
	
		Query<Integer,Integer> q1 = Q.sWindowNaive(3, 0, Integer::sum);
		SLastCount<Integer> sink1 = S.lastCount();

		Query<Integer,Integer> q2 = Q.sWindow3((x, y, z) -> x + y + z);
		SLastCount<Integer> sink2 = S.lastCount();
		
		int n = 1000;
		Iterator<Integer> it = Q.intStream(n);
		q1.start(sink1);
		q2.start(sink2);
		while (it.hasNext()) {
			Integer item = it.next();
			q1.next(item, sink1);
			q2.next(item, sink2);
			assertEquals(sink1.count, sink2.count);
			assertEquals(sink1.last, sink2.last);
		}
		q1.end(sink1);
		q2.end(sink2);
		assertEquals(sink1.count, sink2.count);
		assertEquals(sink1.last, sink2.last);
	}

	@Test
	public void testWindow() {
		System.out.println("***** Test Window (Naive & Efficient) *****");
	
		int wMax = 10;
		for (int w=1; w<wMax; w++) {
			Query<Integer,Integer> q1 = Q.sWindowNaive(w, 0, Integer::sum);
			SLastCount<Integer> sink1 = S.lastCount();

			Query<Integer,Integer> q2 =
				Q.sWindowInv(w, 0, Integer::sum, (x,y) -> x-y);
			SLastCount<Integer> sink2 = S.lastCount();
			
			int n = 20;
			Iterator<Integer> it = Q.intStream(n);
			q1.start(sink1);
			q2.start(sink2);
			while (it.hasNext()) {
				Integer item = it.next();
				q1.next(item, sink1);
				q2.next(item, sink2);
				assertEquals(sink1.count, sink2.count);
				assertEquals(sink1.last, sink2.last);
			}
			q1.end(sink1);
			q2.end(sink2);
			assertEquals(sink1.count, sink2.count);
			assertEquals(sink1.last, sink2.last);
		}
	}

}