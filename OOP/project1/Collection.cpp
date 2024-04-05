#include <iostream>
using namespace std;

class Collection {
	int c[20];
public:
	Collection() { // initialize the array elements of c to 1
		for (int i = 0; i < 20; i++)
			c[i] = 1;
	} 

	Collection(int* x) { // initialize array c using array x
		for (int i = 0; i < 20; i++)
			c[i] = x[i];
	}

	void print() {
		for (int i = 0; i < 20; i++)
			cout << c[i] << " ";
	}

	Collection operator-(const Collection& a) {
		Collection temp; //temp object to store the result

		for (int i = 0; i < 20; i++) {
			temp.c[i] = c[i] - a.c[i]; //c[i] the object that called the function - a.c[i] the object the function recieved 
		}

		return temp;
	}

	bool operator>(const Collection& a) {
		int sum1 = 0, sum2 = 0;

		for (int i = 0; i < 20; i++) {
			sum1 += c[i]; //the object that called the function
			sum2 += a.c[i]; //the object the function recieved 
		}

		return sum1 > sum2; 
	}

};
