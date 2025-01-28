#include <vector>
#include <iostream>
using namespace std;

void mergeSort(vector<int>& a,int first,int last,long long &p);
void merge(vector<int>& a,int first,int mid,int last,long long &p);

void mergeSort(vector<int>& a,int first,int last,long long &p){
    if (first >= last)
        return;

    int mid = first + (last - first) /2;

    mergeSort(a,first,mid,p);
    mergeSort(a,mid+1,last,p);

    merge(a,first,mid,last,p);    
}

void merge(vector<int>& a,int first,int mid,int last,long long &p){
    int sub1 = mid - first + 1;
    int sub2 = last - mid;

    vector<int> leftSub(sub1);
    vector<int> rightSub(sub2);

    for(int i=0; i<sub1; i++)
        leftSub[i] = a[first+i];

    for(int i=0; i<sub2; i++)
        rightSub[i] = a[mid+i+1];  

    int i = 0;
    int j = 0;
    int k = first;

    while(i < sub1 && j < sub2){
        if(leftSub[i] <= rightSub[j]){
            a[k] = leftSub[i++];
        } else {
            a[k] = rightSub[j++];
            p+=(sub1 - i);
        }
        k++;
    }

    while(i < sub1){
        a[k] = leftSub[i++];
        k++;
    }

    while(j < sub2){
        a[k] = rightSub[j++];
        k++;
    }
}

int main() {

    vector<int> ranking; 
    int rank;

    while(cin >> rank)
        ranking.push_back(rank);

    int size = ranking.size();
    long long penalty = 0; 

    mergeSort(ranking,0,size-1,penalty);
    
    cout << penalty;

    return 0;
}

