package main

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"
)

func main() {
	reader := bufio.NewReader(os.Stdin)

	num, err := strconv.Atoi(readLine(reader))
	if err != nil {
		panic(err)
	}

	for i := 0; i < num; i++ {
		works(reader)
	}
}

func works(reader *bufio.Reader) {
	n, err := strconv.Atoi(readLine(reader))
	if err != nil {
		panic(err)
	}

	s := readLine(reader)
	if len(s) > n {
		panic("value error")
	}

	groups := make(map[rune]int)
	for _, c := range s {
		groups[c]++
	}

	counti := 0
	for _, count := range groups {
		if count > 1 {
			counti = count
			break
		}
	}

	fmt.Println(counti)
}

func readLine(reader *bufio.Reader) string {
	line, _ := reader.ReadString('\n')
	return strings.TrimSpace(line)
}
