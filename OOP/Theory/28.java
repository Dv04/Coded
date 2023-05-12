package Theory.TwentyThree.Three;


interface Employee{
    void getData(int empId, String empName);
}

class DemoClass implements Employee{
    public void getData(int empId, String empName){
        System.out.println("Employee Id: "+empId);
        System.out.println("Employee Name: "+empName);
    }
    public static void main(String[] args){
        DemoClass obj = new DemoClass();
        obj.getData(1001, "John");
    }
}