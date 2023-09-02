package main

import  "fmt"

func main() {
 n := 2
 
 // "make" the channel, which can be used
 // to move the int datatype
 out := make(chan int)

 // run this function as a goroutine
 // the channel that we made is also provided
 go Square(n, out)

 // Any output is received on this channel
 // print it to the console and proceed
 fmt.Println(<-out)
}

func Square(n int, out chan<- int) {
 result := n * n
 
 //pipes the result into it
 out <- result
}