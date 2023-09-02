// Given an integer x, return true if x is a palindrome, and false otherwise.
package main

func isPalindrome(x int) bool {
	if x < 0 {
		return false
	}
	if x < 10 {
		return true
	}
	if x%10 == 0 {
		return false
	}
	var y int
	for x > y {
		y = y*10 + x%10
		x /= 10
	}
	return x == y || x == y/10
}

func main() {
	println(isPalindrome(121))
	println(isPalindrome(-121))
	println(isPalindrome(10))
}
