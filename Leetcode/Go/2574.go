package main

import "fmt"

func leftRigthDifference(nums []int) []int {
    res := make([]int, len(nums))
    var rsum, lsum int

    for _, v := range nums {
        rsum += v
    }

    for i, v := range nums{
        rsum -= v
        res[i] = abs(lsum - rsum)
        lsum += v
    }

    return res
}

func abs(n int) int {
    if n < 0 {
        return -n
    }
    return n
}

func main() {
	nums := []int{2, 7, 11, 15}
	fmt.Println(nums)
	fmt.Print(leftRigthDifference(nums))
}