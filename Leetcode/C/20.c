// Given a string s containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.
// An input string is valid if:
// Open brackets must be closed by the same type of brackets.
// Open brackets must be closed in the correct order.
// Every close bracket has a corresponding open bracket of the same type.

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>

bool isValid(char * s){
    int len = strlen(s);
    if (len % 2 != 0) {
        return false;
    }
    char *stack = (char *)malloc(len);
    int top = -1;
    for (int i = 0; i < len; i++) {
        if (s[i] == '(' || s[i] == '[' || s[i] == '{') {
            stack[++top] = s[i];
        } else {
            if (top == -1) {
                return false;
            }
            if (s[i] == ')' && stack[top] != '(') {
                return false;
            }
            if (s[i] == ']' && stack[top] != '[') {
                return false;
            }
            if (s[i] == '}' && stack[top] != '{') {
                return false;
            }
            top--;
        }
    }
    if (top != -1) {
        return false;
    }
    return true;
}

int main() {
    char *s = "()[]";
    if (isValid(s)) {
        printf("true\n");
        } else {
        printf("false\n");
    }
    return 0;
}