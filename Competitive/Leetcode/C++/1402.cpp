#include <bits/stdc++.h>

using namespace std;

class Solution {
public:
    int maxSatisfaction(vector<int>& satisfaction) {
        int ans = 0;
        sort(satisfaction.begin(), satisfaction.end());
        reverse(satisfaction.begin(), satisfaction.end());

        int s = 0;
        int dishSum = 0;

        for (int dish : satisfaction) {
            dishSum += dish;
            if (dishSum <= 0) {
                break;
            }
            s += dishSum;
        }

        return ans;
    }
};


// class Solution:
//     def maxSatisfaction(self, satisfaction: List[int]) -> int:
//         satisfaction.sort(reverse=True)
//         s = dishSum = 0
//         for dish in satisfaction:
//             dishSum += dish
//             if dishSum <= 0:
//                 break
//             s += dishSum
//         return s