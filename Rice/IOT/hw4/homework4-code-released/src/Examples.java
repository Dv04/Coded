import dsl.*;

public class Examples {
	
	public static void main(String[] args) {
		System.out.println("*****************************************************");
		System.out.println("***** ToyDSL: Dataflow Domain-Specific Language *****");
		System.out.println("*****************************************************");
		System.out.println();

		// stream length
		int n = 10 * 1000 * 1000;

		System.out.println("***** map *****");
		{
			Query<Integer,Integer> q = Q.map(x -> 2*x);
			Q.execute(Q.intStream(n), q, S.lastCount());
		}
		System.out.println();

		System.out.println("***** filter *****");
		{
			Query<Integer,Integer> q = Q.filter(x -> x % 2 == 0);
			Q.execute(Q.intStream(n), q, S.lastCount());
		}
		System.out.println();

		System.out.println("***** fold *****");
		{
			Query<Integer,Long> q = Q.fold(0L, (x, y) -> x + y);
			Q.execute(Q.intStream(n), q, S.lastCount());
		}
		System.out.println();

		System.out.println("***** fold avg *****");
		{
			Query<Integer,Double> q =
				Q.pipeline(
					Q.map(x -> Double.valueOf(x)),
					Q.foldAvg()
				);
			Q.execute(Q.intStream(n), q, S.lastCount());
		}
		System.out.println();

		System.out.println("***** scan *****");
		{
			Query<Integer,Long> q = Q.scan(0L, (x, y) -> x + y);
			Q.execute(Q.intStream(n), q, S.lastCount());
		}
		System.out.println();

		System.out.println("***** sWindowNaive *****");
		{
			Query<Integer,Integer> q = Q.sWindowNaive(10, 0, (x, y) -> x + y);
			Q.execute(Q.intStream(n), q, S.lastCount());
		}
		System.out.println();

		System.out.println("***** sWindowInv *****");
		{
			Query<Integer,Integer> q =
				Q.sWindowInv(10, 0, (x, y) -> x + y, (x, y) -> x - y);
			Q.execute(Q.intStream(n), q, S.lastCount());
		}
		System.out.println();

		System.out.println("***** sWindow2 *****");
		{
			Query<Integer,Integer> q = Q.sWindow2((x, y) -> y - x);
			Q.execute(Q.intStream(n), q, S.lastCount());
		}
		System.out.println();

		System.out.println("***** map >> map *****");
		{
			Query<Integer,Integer> q1 = Q.map(x -> 2*x);
			Query<Integer,Integer> q2 = Q.map(x -> 2*x);
			Query<Integer,Integer> q = Q.pipeline(q1, q2);
			Q.execute(Q.intStream(n), q, S.lastCount());
		}
		System.out.println();

		System.out.println("***** filter >> map >> fold *****");
		{
			Query<Integer,Integer> q1 = Q.filter(x -> x % 2 == 0);
			Query<Integer,Integer> q2 = Q.map(x -> 2*x);
			Query<Integer,Long> q3 = Q.fold(0L, (x, y) -> x + y);
			Query<Integer,Long> q = Q.pipeline(q1, q2, q3);
			Q.execute(Q.intStream(n), q, S.lastCount());
		}
		System.out.println();
	}

}
