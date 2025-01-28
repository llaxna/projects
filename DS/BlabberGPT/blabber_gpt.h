#include "stack.h"
#include <iostream>
using namespace std;

class BlabberGPT {
private:
    // ... add private members
    string txt;
    int size;
    Map<string,Stack<char>> m;

public:
    // The only constructor.
    // @param text: The text to be used in building the model
    // @param    k: The substring size in model
    //
    // This function must perform O(nlogn) string compares
    // for typical English text, where n is the number of 
    // characters in the parameter "text".
    BlabberGPT(string& text, int k) {
        // .. add your implementation
        if(k<=0)
            throw string("invalid");

        txt = text;
        size = k;
        int textLength = txt.length();

        /////using built in function substr
        /*
        for(int i=0; i< textLength-size ; i++){
            string substring = txt.substr(i,size);
            char next = txt[i+size];

            if(!m.contains(substring)){
                Stack<char> s;
                m.insert(substring,s);
            }
            
            m.value_of(substring).push(next);
          
        }
        */
        
        for(int i=0;i<textLength-size;i++){
            string sub;
            for(int j=i;j<i+size ;j++)
                 sub += txt[j];

            char next = txt[i+size];

            if(!m.contains(sub)){
                Stack<char> s;
                m.insert(sub,s);
            }
            
            m.value_of(sub).push(next);
               
        }
    }

    // Prints the model in the following format:
    //
    // substring1: [char1, char2, char3, etc.]
    // substring2: [char1, char2, etc.]
    // ...
    //
    // The substrings must be ordered alphabatically in ascending order.
    // The characters associated with a substring must appear in the
    // same order of their appearance in the text.
    // See the examples provided in the description of step 1.
    void print() {               
        // .. add your implementation
        cout << m;
    }

    // Generates random text using the model
    //
    // @param   prompt:  A string of size k to begin the generation from.
    // @param        n:  The number of characters to generate.
    // @throw   string:  If "n" is negative, or if "prompt" is not of size k
    //                   or if "prompt" does not appear in the text that
    //                   was used for creating the model.
    //
    // This function must perform O(nlogN) string compares
    // for typical English text, where n is the number of 
    // characters to be generated and N is the number of strings
    // in the model.
    void generate(int n, string prompt) {
        // .. add your implementation
        int promptLength = prompt.length();
        
        if((n<0) || (promptLength != size) || (!m.contains(prompt)))
            throw string("ERROR !!");

        string str = prompt;

        for(int i=0;i<n;i++){
            string curr;
            for(int j=i;j<i+promptLength ;j++)
                 curr += str[j];

            Stack<char> s = m.value_of(curr);

            if(s.is_empty())
                break;

            char randomC = s.sample();
            str += randomC;         

        }         

        cout << str << endl;


        ////using substr function
/*
        for(int i=0;i<n;i++){
            string curr = str.substr(str.length()-promptLength,promptLength);
            Stack<char> s = m.value_of(curr);

            if(s.is_empty())
                break;

            char randomC = s.sample();
            str += randomC;    
                    
        }
        
    cout << str << endl;
*/
    }
};

