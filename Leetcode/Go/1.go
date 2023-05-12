package main

import "fmt"

func twoSum(nums []int, target int) []int {

	var result []int
	var m = make(map[int]int)

	for i, v := range nums {
		if _, ok := m[target-v]; ok {
			result = append(result, m[target-v])
			result = append(result, i)
			return result
		}
		m[v] = i
	}
	fmt.Print("1")
	fmt.Println(result)
	return result

}

// func main() {
// 	nums := []int{2, 7, 11, 15}
// 	target := 9
// 	twoSum(nums, target)
// }
