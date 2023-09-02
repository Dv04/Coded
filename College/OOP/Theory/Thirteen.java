class Table {
    public int x;

    synchronized void printTable(int n, Table t, int m) {
        for (int i = 1; i <= 5; i++) {
            t.x = t.x * m;
            System.out.println(t.x * m);
            try {
                Thread.sleep(400);
            } catch (Exception e) {
                System.out.println(e);
            }
        }
    }
}

class MyThread1 extends Thread {
    Table t;

    MyThread1(Table t) {
        this.t = t;
    }

    public void run() {
        t.printTable(50, t, 1);
    }
}

class MyThread2 extends Thread {
    Table t;

    MyThread2(Table t) {
        this.t = t;
    }

    public void run() {
        t.printTable(100, t, 2);
    }
}

class Thirteen {
    public static void main(String args[]) {
        Table obj = new Table();
        obj.x = 50;
        MyThread1 t1 = new MyThread1(obj);
        MyThread2 t2 = new MyThread2(obj);
        t1.start();
        t2.start();
    }
}