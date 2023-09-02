// package Theory;

import java.util.Scanner;

public class Main {
    
    
    public static void main(String[] args) {
        System.out.println("Hello World");

        Test1 t1 = new Test1();
        t1.test1(1, "test");
    }
    
}

class Test1 {
    Test1() {
        try (Scanner sc = new Scanner(System.in)) {
            int id = sc.nextInt();
            String name = sc.next();
            test1(id, name);
        }
    }
    void test1 (int id, String name) {
        System.out.println(id + " " + name);
    }
}