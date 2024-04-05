#include <iostream>
#include "Collection.cpp"
using namespace std;

int main() {
	int x[] = { 1,4,3,8,5,6,10,8,9,20,7,4,8,4,5,6,7,8,9,12 };
	int y[] = { 1,2,3,4,5,6,7,8,9,10,6,3,3,4,1,3,7,8,5,11 };

	Collection A(x), B(y), C;

	//operator>
	if (A > B) //A.operator>(B)
		cout << "A is greater than B" << endl;
	else
		cout << "A is not greater than B" << endl;

	C.print();

	//operator- 
	C = A - B; //A.operator-(B)

	C.print();
	
	return 0;
}