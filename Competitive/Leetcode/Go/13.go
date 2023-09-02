// Given a roman numeral, convert it to an integer.

package main

func romanToInt(s string) int {
	var x int
	for i := 0; i < len(s); i++ {
		switch s[i] {
		case 'I':
			x++
		case 'V':
			x += 5
			if i > 0 && s[i-1] == 'I' {
				x -= 2
			}
		case 'X':
			x += 10
			if i > 0 && s[i-1] == 'I' {
				x -= 2
			}
		case 'L':
			x += 50
			if i > 0 && s[i-1] == 'X' {
				x -= 20
			}
		case 'C':
			x += 100
			if i > 0 && s[i-1] == 'X' {
				x -= 20
			}
		case 'D':
			x += 500
			if i > 0 && s[i-1] == 'C' {
				x -= 200
			}
		case 'M':
			x += 1000
			if i > 0 && s[i-1] == 'C' {
				x -= 200
			}
		}
	}
	return x
}

func main() {
	println(romanToInt("III"))
	println(romanToInt("IV"))
	println(romanToInt("IX"))
	println(romanToInt("LVIII"))
	println(romanToInt("MCMXCIV"))
}
