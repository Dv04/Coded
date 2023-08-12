package main

import "fmt"

func main() {
	var a int
	fmt.Print("Enter the size: ")
	fmt.Scan(&a)
	for i := 0; i < a; i++ {
		for j := 0; j <= i; j++ {
			fmt.Print("*")
		}
		fmt.Println()
	}
}
