// This is a code for class object in Javascript

class student { 
    constructor(name, age, grade) {
        this.name = name;
        this.age = age;
        this.grade = grade;
    }
    setName(name) {
        this.name = name;
    }
    setAge(age) {
        this.age = age;
    }
    setGrade(grade) {
        this.grade = grade;
    }
    getAge() {
        return this.age;
    }
    getName() {
        return this.name;
    }
    getGrade() {
        return this.grade;
    }
}

let student1 = new student();
student1.setName("Rahul");
student1.setAge(20);
student1.setGrade("A");
let student2 = new student("Raj", 20, "A+");

console.log(student1.getName());
console.log(student2.getName());