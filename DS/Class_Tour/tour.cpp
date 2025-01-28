#include "point.h"
#include <iostream>
using namespace std;


template <class T>
class Node {
public:
    T val;
    Node* next;
    Node* prev;

    Node(const T& val, Node* next, Node* prev) {
        this->val = val;
        this->next = next;
        this->prev = prev;
    }
};


template <class T>
class Tour {
private:
    Node<T>* start;
    int size;

    // inserts a new node with the given value after
    // the given node.
    // Throws a string exception if node is a nullptr
    void insert_after(Node<T>* node, const T& val);

    // inserts a new node with the given value before
    // the given node.
    // Throws a string exception if node is a nullptr
    void insert_before(Node<T>* node, const T& val);
    
    //check if the list is empty
    bool is_empty() const;

public:
    Tour() { start = nullptr; size = 0; }
    ~Tour() { clear(); }
   
    Node<T>* get_start() const { return start; }
    int get_size() const { return size; }
    
    // Adds a new node with the given value.
    // The newly added node becomes the last node in the tour.
    // (the node before start)
    void add_to_end(const T&);
    
    // Adds a new node with the given value.
    // The newly added node becomes the start of the tour.
    // The old start node becomes the 2nd node in the tour.
    void add_to_start(const T&);

    // Removes all the nodes in the tour between ptr1 and ptr2 (exclusive).
    // - If "start" is in the cut range, start points at ptr1.
    // - If ptr1 == ptr2, all nodes in the tour are removed except ptr1/ptr2
    //   and start points at the remaining node.
    // - Throws a string exception if ptr1 or ptr2 is a nullptr.
    // - Throws a string exception if ptr1 and ptr2 point to nodes 
    //   that are not in the same tour.
    void cut(Node<T>* ptr1, Node<T>* ptr2);

    // Removes all the nodes in the tour.
    // (This function must use function cut());
    void clear();

    // Check if the tour is symmetric. I.e. if visiting 
    // the nodes from start node to the last node (the right before the start node) 
    // produces the same sequence of "values" for visiting the nodes starting from 
    // the last one and back to the start node.
    // Consider an empty tour and a tour with one node as symmetric.
    // Example.
    // the following are symmetric tours (of Point2D objects): 
    //    (1, 2) --> (2, 5) --> (1, 2) 
    //    (1, 2) --> (2, 5) --> (5, 8) --> (2, 5) --> (1, 2)
    //    (1, 2) --> (2, 5) --> (5, 8) --> (5, 8) --> (2, 5) --> (1, 2)
    // the following are not symmetric tours:
    //    (1, 2) --> (2, 5)
    //    (1, 2) --> (2, 5) --> (5, 8) --> (2, 5)
    //    (1, 2) --> (2, 5) --> (5, 8) --> (4, 6) --> (2, 5) --> (1, 2)
    bool is_symmetric_tour() const;

    // Computes the length of a tour of Point2D objects.
    // The length of the tour is defined as:
    // length = dist(start_node, 2nd_node) + dist(2nd_node, 3rd_node) + ... +
    //          dist(last_node,  start_node)
    // Example.
    //    (0, 0) --> (1, 1) --> (2, 2) --> (3, 3)
    //    length = dist((0,0), (1,1)) + dist((1,1), (2,2)) +
    //             dist((2,2), (3,3)) + dist((3,3), (0,0))
    //           =  sqrt(2) + sqrt(2) + sqrt(2) + sqrt(18)
    //           =  8.48528137424
    // Return 0 if the tour is empty or contains 1 point.
    friend double tour_length(const Tour<Point2D>& tour);

    // Prints the values in the tour.
    // -- DO NOT IMPLEMENT THIS FUNCTION. YOU CAN USE IT IMMEDIATELY.
    // -- THIS FUNCTION IS GIVEN TO YOU BY THE ENVIRONMENT. 
    // -- IF YOU IMPLEMENT IT, YOU WILL GET COMPILATION ERRORS.
    template<class K> 
    friend ostream& operator<<(ostream& out, const Tour<K>& tour);
};

template <class T>
void Tour<T>::insert_after(Node<T>* node, const T& val){
    if(node == nullptr || is_empty())
        throw string("the node is empty");    

    Node<T>* next_node = node->next;
    Node<T>* new_node = new Node<T>(val, next_node, node);
    node->next = new_node;
    next_node->prev = new_node;
    size++;
}

template <class T>
void Tour<T>::insert_before(Node<T>* node, const T& val){
    if(node == nullptr || is_empty())
        throw string("the node is empty");

    Node<T>* pred = node->prev;     
    Node<T>* new_node = new Node<T>(val, node, pred);
    node->prev = new_node;
    pred->next = new_node;
    size++;
}

template <class T>
bool Tour<T>::is_empty() const { return start == nullptr;}

template <class T>
void Tour<T>::add_to_end(const T& val){
    if(is_empty()){
        start = new Node<T>(val,nullptr,nullptr);
        start->prev = start;
        start->next = start;
        size++;
        return;
    }
    
   Node<T>* pred = start->prev;
   insert_after(pred,val);
}

template <class T>
void Tour<T>::add_to_start(const T& val){
    if(is_empty()){
        start = new Node<T>(val,nullptr,nullptr);
        start -> prev = start; 
        start -> next = start;
        size++;
        return;
    }

    Node<T>* curr = start;
    insert_before(curr,val);
    start = curr->prev;
}

template<class T>
void Tour<T>::cut(Node<T>* ptr1, Node<T>* ptr2){
    if( is_empty() || (ptr1==nullptr || ptr2==nullptr))
        throw string("can't remove the nodes");

    Node<T>* check = ptr1;
    while(check!=ptr2 && check!=ptr1->prev)
        check = check->next;

    if(check!=ptr2 && ptr1 != ptr2)
        throw string("the pointers are not in the same tour");

    Node <T>* curr = ptr1->next;
    while(curr!=ptr2){
        curr = curr ->next;
        delete curr->prev;
        size--;

        if(curr==start)
            start = ptr1;
    }

    ptr1->next = ptr2;
    ptr2->prev = ptr1;
}

template<class T>
void Tour<T>::clear(){
     if(is_empty())
         return;

     cut(start,start);
     start = nullptr;
     size=0;
}

template<class T>
bool Tour<T>:: is_symmetric_tour() const{
    if(is_empty() || size==1)
         return true;
    
    Node<T> *s = start;
    Node<T> *end = start->prev;

    if(end->prev == s)
        if(s->val != end->val)
          return false;

    while(end->prev != s){
        if(s->val != end->val)
           return false;

        s = s->next;  
        end = end->prev;
    }
    return true;
}

double tour_length(const Tour<Point2D>& tour) {
    // ADD YOUR IMPLEMENTATION HERE
    if(tour.size==0 || tour.size==1)
        return 0;

    double length = 0;
    Node<Point2D>* st = tour.start;
    Node<Point2D>* nxt = tour.start->next;

    while(nxt!=tour.start){
        length+=st->val.distance_to(nxt->val);
        st = st->next;
        nxt = nxt->next;
    }

    //to calculate the first node with the last
    length+=st->val.distance_to(nxt->val);

    return length;            
}


// ****** Do NOT add a main function here
// ****** Use main.cpp instead.


