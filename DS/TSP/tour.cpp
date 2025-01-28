#include "point2d.h"
#include <iostream>
#include <fstream>
#include "cturtle/CImg.h"
#include "cturtle/CTurtle.hpp"

using namespace std;
using namespace cturtle;

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
public:
    Tour() { start = nullptr; size = 0; }
    ~Tour() { clear(); }

    Node<T>* get_start() const { return start; }
    int get_size() const { return size; }

    void add_to_end(const T&);
    void add_to_start(const T&);

    void cut(Node<T>* ptr1, Node<T>* ptr2);
    void clear();

private:
    bool is_empty() const;
    void insert_after(Node<T>* node, const T& val);
    void insert_before(Node<T>* node, const T& val);

    template<class K> friend ostream& operator<<(ostream& out, const Tour<K>& tour);
    friend double tour_length(const Tour<Point2D>& tour);

    friend void insert_closest(const Point2D& point, Tour<Point2D>& tour);
    friend void insert_smallest(const Point2D& point, Tour<Point2D>& tour);

private:
    int size;
    Node<T>* start;
};


//////////////////////////////////////////////////////////////////////
///////// COPY YOUR FUNCTION IMPLEMENTATATIONs FROM EX1 HERE /////////
//////////////////////////////////////////////////////////////////////
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

void insert_closest(const Point2D& point, Tour<Point2D>& tour) {
    // IMPLEMENT THIS FUNCTION
    if(tour.size==0){
        tour.add_to_start(point);
        return;
    }
    
    Node<Point2D>* closest= tour.start;
    Node<Point2D>* curr= closest->next;
    double min_dist = point.distance_to(closest->val);

    while(curr!=tour.start){
      double dist = point.distance_to(curr->val);
        if (dist<min_dist){
            min_dist = dist;
            closest = curr;
        }

        curr=curr->next;    
    }

    tour.insert_after(closest,point);
}

void insert_smallest(const Point2D& point, Tour<Point2D>& tour) {
    // IMPLEMENT THIS FUNCTION
     if(tour.size==0){
        tour.add_to_start(point);
        return;
    }
    
    Node<Point2D>* smallest= tour.start;
    Node<Point2D>* curr= smallest->next;

    double tour_len = tour_length(tour); //calculates the length of the tour before adding the point
    double min_length = tour_len + smallest->val.distance_to(point)+point.distance_to(curr->val)-curr->val.distance_to(smallest->val);
    //calculates the length of the tour when adding the point after the first point

   while(curr!=tour.start){
        double length = tour_len + curr->val.distance_to(point)+point.distance_to(curr->next->val)-curr->next->val.distance_to(curr->val);
        //calculates the length of the tour when adding the point after each point in the tour

        if(length<min_length){
            min_length=length;
            smallest = curr;
        }

        curr = curr -> next;
    
    }

    tour.insert_after(smallest,point);
}



////////////////////////////////////////////////////////////
/////////// DO NOT CHANGE WHAT'S UNDER THIS LINE ///////////
////////////////////////////////////////////////////////////

void draw(const Tour<Point2D>&, int, int);

int main() {
    string type;
    while (true) {
        cout << "Choose strategy: c (closest point) s (smallest increase): " << endl;
        cin >> type;
        if (type == "c" || type == "s")
            break;
        else
            cout << "ERROR: INVALID INPUT!" << endl;
    }
    
    string filename;
    ifstream fin;
    do {
        cout << "Enter the name of the test file: ";
        cin >> filename;
        fin.open("files/" + filename);
        if (!fin)
            cout << "INVALID FILE NAME\n";
    } while (!fin);

    Tour<Point2D> tour;

    double width, height;
    fin >> width >> height;
    width += 20;
    height += 20;

    double x, y;
    cout << "Working ... " << endl;
    while (fin >> x >> y) {
        if (type == "c")
            insert_closest(Point2D(x, y), tour);
        else
            insert_smallest(Point2D(x, y), tour);
    }
    
    cout << endl;
    cout << "Tour: " << endl << tour 
         << "Length = " << tour_length(tour) << endl
         << "Size = " << tour.get_size() << " points" << endl;

    draw(tour, width, height);

    return 0;
}


void draw(const Tour<Point2D>& tour, int w, int h) {
    int size = tour.get_size();
    if (size == 0)
        return;

    Node<Point2D>* curr = tour.get_start();
    TurtleScreen scr;
    Turtle turtle(scr);

    // go to the first position
    turtle.penup();
    turtle.goTo(-scr.window_width()/2 + (curr->val.x / w) * (scr.window_width()), 
                -scr.window_height()/2 + (curr->val.y / h) * (scr.window_height()));
    turtle.pendown();
    turtle.speed(TS_FASTEST);

    if (size > 10000)
        scr.tracer(size / 10000, 0);

    // turtle.penup();
    for (int i = 0; i <= size; i++, curr = curr->next) {
        Point2D point = curr->val;
        int new_x = -scr.window_width()/2 + (point.x / w) * (scr.window_width());
        int new_y = -scr.window_height()/2 + (point.y / h) * (scr.window_height());
        
        turtle.goTo(new_x, new_y);
    }

    scr.mainloop();
}


