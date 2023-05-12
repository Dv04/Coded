package main

import (
	"fmt"
)

func main() {

	// Assigning an anonymous
	// function to a variable
	value := func() {
		fmt.Print("Welcome! to GeeksforGeeks\n")
	}
	value()
	valu()
}

func valu() {
	fmt.Println("Welcome! to GeeksforGeeks")
}
