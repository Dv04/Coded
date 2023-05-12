// You are given a non-negative floating point number rounded to two decimal places celsius, that denotes the temperature in Celsius.

// You should convert Celsius into Kelvin and Fahrenheit and return it as an array ans = [kelvin, fahrenheit].

#include <bits/stdc++.h>

using namespace std;

class Solution {
public:
    vector<double> convertTemperature(double celsius) {
        vector<double> ans;
        ans.push_back(celsius + 273.15);
        ans.push_back(celsius * 1.80 + 32.00);
        return ans;
    }
};