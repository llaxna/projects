#include "tour.cpp"

// DO NOT IMPLEMENT IN THIS FILE ANY FUNCTION 
// FROM CLASS TOUR. THIS FILE IS ONLY FOR TESTING
// CLASS TOUR.

int main() {
    Tour<int> myList;
    Tour<Point2D> tour2D;

    cout << "Testing add_to_start:" << endl;
    myList.add_to_start(2);
    myList.add_to_start(1);
    cout << myList << endl;

    cout << "--------------------------" <<endl;
    cout << "Testing add_to_end:" << endl;
    myList.add_to_end(1);
    cout << myList << endl;

    cout << "--------------------------" <<endl;
    cout << "Testing is_symmetric_tour:" << endl;
    bool sym = myList.is_symmetric_tour();

    if (sym)
         cout << "Tour is symmetric:" << myList << endl;
    else
         cout << "Tour is not symmetric:" << myList << endl;

    cout << "--------------------------" <<endl;
    cout << "Testing tour_length:" << endl;

    tour2D.add_to_start(Point2D(1,3));
    tour2D.add_to_start(Point2D(2,4));
    tour2D.add_to_start(Point2D(5,6));
    tour2D.add_to_start(Point2D(2,4));
    tour2D.add_to_start(Point2D(3,5));

    const Tour<Point2D> myList2(tour2D);
    double tourLength = tour_length(myList2);
    cout << "the length of the tour is: " << tourLength << endl;   
    cout << myList2 << endl;

    cout << "--------------------------" <<endl;
    cout << "Testing cut:" << endl;

    Node<Point2D>* start = tour2D.get_start()->next;
    Node<Point2D>* end = tour2D.get_start()->prev;
    tour2D.cut(start,end); 
    cout << tour2D <<endl;    

    cout << "--------------------------" <<endl;
    cout << "Testing clear:" << endl;
    myList.clear();
    cout << myList << endl;

    return 0;
}
