#pragma once
#include <ostream>
#include <iostream>
using namespace std;

const int DEFAULT_CAPACITY = 8;

template <class T>
class Stack
{
public:
    Stack();
    Stack(const Stack& other);
    ~Stack();

    void clear();
    void push(const T& val);
       T pop();
       T sample()   const;
     int size()     const { return last + 1; }
    bool is_empty() const { return last == -1; }

    Stack& operator=(const Stack& other);
    template <class K> friend ostream& operator<<(ostream&, const Stack<K>&);

private:
    T* data;
    int capacity;
    int last;

    void resize(int new_cap);
};


template <class T>
Stack<T>::Stack()
{
    capacity = DEFAULT_CAPACITY;
    data = new T[capacity];
    last = -1;
}

template <class T>
Stack<T>& Stack<T>::operator=(const Stack<T>& other) {
    if (this == &other)
        return *this;

    delete [] data;
    capacity = other.capacity;
    data = new T[capacity];

    last = other.last;
    for (int i = 0; i <= last; i++)
        data[i] = other.data[i];

    return *this;
}

template <class T>
Stack<T>::Stack(const Stack<T>& other) {
    capacity = other.capacity;
    data = new T[capacity];

    last = other.last;
    for (int i = 0; i <= last; i++)
        data[i] = other.data[i];
}

template <class T>
Stack<T>::~Stack()
{
    delete [] data;
}

template <class T>
void Stack<T>::push(const T& val)
{
    if (last == capacity - 1)
        resize(capacity * 2);

    data[++last] = val;
}

template <class T>
T Stack<T>::pop()
{
    if (is_empty())
        throw string("ERROR: Stack underflow!");
        
    T val = data[last--];

    if (last > DEFAULT_CAPACITY && last < capacity / 4)
        resize(capacity / 2);
        
    return val;
}

template <class T>
T Stack<T>::sample() const
{
    if (is_empty()) 
        throw string("ERROR: Attempting to retrieve an element from an empty stack!");

    return data[rand() % (last + 1)];
}

template <class T>
void Stack<T>::clear()
{
    last  = -1;
    delete [] data;
    data = new T[DEFAULT_CAPACITY];
    capacity = DEFAULT_CAPACITY;
}

template <class T>
void Stack<T>::resize(int new_cap) 
{
    if (new_cap <= last)
        throw string("Invalid new capacity in resize(...)");

    T* new_data = new T[new_cap];
    for (int i = 0; i <= last; i++)
        new_data[i] = data[i];

    delete [] data;
    data = new_data;
    capacity = new_cap;
}

template <class T>
ostream& operator<<(ostream& out, const Stack<T>& stack) {
    out << "[";
    for (int i = 0; i <= stack.last; i++) {
        out << stack.data[i];
        if (i < stack.last) out << ", ";
    }
    out << "]";
    return out;
}
