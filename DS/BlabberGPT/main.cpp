#include <sstream>
#include <fstream>
#include "blabber_gpt.h"
using namespace std;

// -- You are not supposed to change the code below -- //
// -------------- Read it if you wish!! -------------- //
int main() {
    string name;
    cout << "Enter file name: ";
    cin >> name;

    int k;
    cout << "Enter substring size: ";
    cin >> k;

    ifstream fin;
    fin.open(name);
    if (!fin) {
        cout << "Invalid file name." << endl;
        return 0;
    }
    stringstream buffer;
    buffer << fin.rdbuf();
    string text = buffer.str();

    cout << "Building model ... " << endl;
    BlabberGPT gpt(text, k);    
    cout << "Done!" << endl;

    cout << "Enter the number of characters to generate: ";
    int n;
    cin >> n;

    cout << "Enter the prompt: ";
    string prompt;
    getchar(); // ignore \n from previous input
    getline(cin, prompt, '\n');
    
    cout << endl << "GENERATED TEXT: " << endl;
    cout << "------------------" << endl;
    try {
        gpt.generate(n, prompt);
    } catch (string s) { cout << s << endl; }
    cout << endl << "------------------" << endl;

    fin.close();
    return 0;
}
