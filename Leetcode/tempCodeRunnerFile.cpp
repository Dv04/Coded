#include <iostream>
#include <vector>
#include <string>

using namespace std;

int main(){
    vector<string> msg{"Hello", "C++", "World", "from", "VS Code", "and the C++ extension!"};

for (vector<string>::const_iterator iter = msg.begin(); iter != msg.end(); ++iter)
{
    const string &word = *iter;
    // Do something with word
    cout << word << " ";
    }

    cout << endl;

    return 0;
}
