// DO NOT CHANGE THIS CLASS 

#include <cmath>
#include <iostream>
using namespace std;

class Point2D {
public:
    double x;
    double y;

    Point2D() {
        x = 0;
        y = 0;
    }

    Point2D(double x, double y) {
        this->x = x;
        this->y = y;
    }

    double distance_to(const Point2D& other) const {
        double diffx = x - other.x;
        double diffy = y - other.y;
        return sqrt(diffx * diffx + diffy * diffy);
    }
};

ostream& operator<<(ostream& out, const Point2D& point) {
    out << "(" + to_string(point.x) + ", " + to_string(point.y) + ")";
    return out;
}
