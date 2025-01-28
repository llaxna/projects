#include <algorithm>
#include <iostream>
#include <chrono>

using namespace std;
using namespace std;
using namespace std::chrono;

// Provide your solution for question 5 here //
void mergeSort(int a[],int n);
void merge(int a[],int first,int mid,int last);

void mergeSort(int a[],int n){

	int size,start,mid,end;

	for(size = 1; size < n; size *= 2){
		for(int i=0; i < n-1; i+=2*size){
			start = i;
			end = i+2*size-1;
			mid = i + size-1;

			if(end>=n)
				end = n-1;
			if(mid>=n)
				mid = n-1;
			merge(a,start,mid,end);
		}
	}

	if(size/2 <= n)
		merge(a,0,(size/2)-1,n-1);
	
}

void merge(int a[], int first, int mid, int last){
	int size = last-first+1;
	int result[size];
	int i = first;
	int j = mid + 1;

	for(int k=0; k<size; k++){
		if(i>mid)
			result[k] = a[j++];
		else if(j>last)
			result[k] = a[i++];
		else if(a[i]<=a[j])
			result[k] = a[i++];
		else
			result[k] = a[j++];	
	}

	for(int m=0;m<size;m++)
		a[m+first] = result[m];
}

void algorithm_B(int a[], int n) {
	mergeSort(a,n);
} 

void algorithm_A(int a[], int n){ //CountingSort
	int min = a[0];
	int max = a[0];

	for(int i=1; i < n; i++){
		if(a[i]<min)
			min = a[i];

		if(a[i]>max)
			max = a[i];	
	}

	int range = max - min;
	int size = range+1;
	int *index = new int[size];

	for(int i=0; i < size; i++)
		index[i]=0;

	for(int i=0; i < n; i++)
		index[a[i]-min]++;
	
	int *sumFreq = new int[size];
	sumFreq[0] = index[0];

	for(int i=1; i < size; i++)
		sumFreq[i] = sumFreq[i-1] + index[i];
	

	int *sortedArray = new int[n];
	for(int i=n-1; i >= 0; i--){
		sortedArray[sumFreq[a[i]-min]-1]=a[i];
		sumFreq[a[i]-min]--;
	}

	for(int i=0;i<n;i++)
		a[i]=sortedArray[i];


	delete[] index;
	delete[] sumFreq;
	delete[] sortedArray;	
}

int partition(int a[],int first,int last){
	int pivot = a[first];
    int i = first;
    int j = last+1;

    while(true){
        do{
            i++;
        }
        while(i < last && a[i] < pivot);

        do{
            j--;
        }
        while(j > first && a[j] > pivot);

        if(i>=j)
            break;
        else
            swap(a[i],a[j]);    
    }

    swap(a[first],a[j]);
    return j;
}

void quick_sort(int a[], int first, int last){
    if(first >= last)
        return;

    int p = partition(a,first,last);
    quick_sort(a,first,p-1);
    quick_sort(a,p+1,last);    
}

void algorithm_C(int a[], int n) {
	for(int i=n-1;i>=0;i--){
		int j = rand()%(i+1);

		swap(a[i],a[j]);
	}

	quick_sort(a,0,n-1);
}



// ----- DO NOT MODIFY THE CODE BELOW THIS ----- //

void printArray(int a[], int n)
{
	for (int i = 0; i < n; i++)
		printf("%d ", a[i]);
	cout << endl;
}

bool is_sorted(int* a, int n) {
	for (int i = 0; i < n-1; i++)
		if (a[i] > a[i+1])
			return false;
	return true;
}

int main()
{
	int n;
	int* rand_array;
	int* sorted_array;
	int* reversly_sorted_array;

	for (int n = 300000; n <= 1200000; n *= 2) {
		cout << endl << " ------- N = " << n << " ------ " << endl;
		rand_array = new int[n];
		sorted_array = new int[n];
		reversly_sorted_array = new int[n];
		
		for (int i = 0; i < n; i++) {
			int x = rand() % (n - 1);
			rand_array[i] = x;
			sorted_array[i] = i;
		}
		for (int i = n - 1, j = 0; i >= 0; i--, j++)
			reversly_sorted_array[j] = i;

		auto start1 = steady_clock::now();
		algorithm_A(sorted_array, n);
		auto stop1 = steady_clock::now();
		auto duration1 = duration_cast<microseconds>(stop1 - start1);

		if (!is_sorted(sorted_array, n))
			cout << "ERROR: ALGORITH A DOES NOT SORT CORRECTLY!" << endl;
		else cout << "Algorithm A (ascending):   " << duration1.count() / 1000.0 << " milliseconds" << endl;

		start1 = steady_clock::now();
		algorithm_A(reversly_sorted_array, n);
		stop1 = steady_clock::now();
		duration1 = duration_cast<microseconds>(stop1 - start1);
		
		if (!is_sorted(reversly_sorted_array, n))
			cout << "ERROR: ALGORITH A DOES NOT SORT CORRECTLY!" << endl;
		else cout << "Algorithm A (descending):  " << duration1.count() / 1000.0 << " milliseconds" << endl;

		start1 = steady_clock::now();
		algorithm_A(rand_array, n);
		stop1 = steady_clock::now();
		duration1 = duration_cast<microseconds>(stop1 - start1);
		
		if (!is_sorted(rand_array, n))
			cout << "ERROR: ALGORITH A DOES NOT SORT CORRECTLY!" << endl;
		else cout << "Algorithm A (random):      " << duration1.count() / 1000.0 << " milliseconds" << endl;
		cout << endl;
		//________________________________________________________

		delete[] rand_array;
		delete[] sorted_array;
		delete[] reversly_sorted_array;

		rand_array = new int[n];
		sorted_array = new int[n];
		reversly_sorted_array = new int[n];

		for (int i = 0; i < n; i++) {
			int x = rand() % (n - 1);
			rand_array[i] = x;
			sorted_array[i] = i;
		}
		for (int i = n - 1, j = 0; i >= 0; i--, j++)
			reversly_sorted_array[j] = i;
		
		start1 = steady_clock::now();
		algorithm_B(sorted_array, n);
		stop1 = steady_clock::now();
		duration1 = duration_cast<microseconds>(stop1 - start1);
		
		if (!is_sorted(sorted_array, n))
			cout << "ERROR: ALGORITH B DOES NOT SORT CORRECTLY!" << endl;
		else cout << "Algorithm B (ascending):   " << duration1.count() / 1000.0 << " milliseconds" << endl;

		start1 = steady_clock::now();
		algorithm_B(reversly_sorted_array, n);
		stop1 = steady_clock::now();
		duration1 = duration_cast<microseconds>(stop1 - start1);
		
		if (!is_sorted(reversly_sorted_array, n))
			cout << "ERROR: ALGORITH B DOES NOT SORT CORRECTLY!" << endl;
		else cout << "Algorithm B (descending):  " << duration1.count() / 1000.0 << " milliseconds" << endl;

		start1 = steady_clock::now();
		algorithm_B(rand_array, n);
		stop1 = steady_clock::now();
		duration1 = duration_cast<microseconds>(stop1 - start1);

		if (!is_sorted(rand_array, n))
			cout << "ERROR: ALGORITH B DOES NOT SORT CORRECTLY!" << endl;
		else cout << "Algorithm B (random):      " << duration1.count() / 1000.0 << " milliseconds" << endl;
		cout << endl;
		//________________________________________________________

		rand_array = new int[n];
		sorted_array = new int[n];
		reversly_sorted_array = new int[n];

		for (int i = 0; i < n; i++) {
			int x = rand() % (n - 1);
			rand_array[i] = x;
			sorted_array[i] = i;
		}
		for (int i = n - 1, j = 0; i >= 0; i--, j++)
			reversly_sorted_array[j] = i;
		
		start1 = steady_clock::now();
		algorithm_C(sorted_array, n);
		stop1 = steady_clock::now();
		duration1 = duration_cast<microseconds>(stop1 - start1);
		
		if (!is_sorted(sorted_array, n))
			cout << "ERROR: ALGORITH C DOES NOT SORT CORRECTLY!" << endl;
		else cout << "Algorithm C (ascending):   " << duration1.count() / 1000.0 << " milliseconds" << endl;

		start1 = steady_clock::now();
		algorithm_C(reversly_sorted_array, n);
		stop1 = steady_clock::now();
		duration1 = duration_cast<microseconds>(stop1 - start1);
		
		if (!is_sorted(reversly_sorted_array, n))
			cout << "ERROR: ALGORITH C DOES NOT SORT CORRECTLY!" << endl;
		else cout << "Algorithm C (descending):  " << duration1.count() / 1000.0 << " milliseconds" << endl;

		start1 = steady_clock::now();
		algorithm_C(rand_array, n);
		stop1 = steady_clock::now();
		duration1 = duration_cast<microseconds>(stop1 - start1);
		
		if (!is_sorted(rand_array, n))
			cout << "ERROR: ALGORITH C DOES NOT SORT CORRECTLY!" << endl;
		else cout << "Algorithm C (random):      " << duration1.count() / 1000.0 << " milliseconds" << endl;

		delete[] rand_array;
		delete[] sorted_array;
		delete[] reversly_sorted_array;
	}
	
	return 0;
}
