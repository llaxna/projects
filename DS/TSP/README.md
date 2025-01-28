# Task Summary
You will implement in this exercise two algorithms (heuristics) for finding approximate solutions for the traveling salesman problem. In both algorithms, the tour is built incrementally by adding the points one by one. Your task is to implement the functions that pick where in the tour newly added points should be inserted.

**Step 1**. Copy your implementation of class Tour from exercise 1 to tour.cpp.

**Step 2**. Implement the following two friend functions. The description of these functions is provided in the following subsection.

void insert_closest(const Point2D& point, Tour<Point2D>& tour);
void insert_smallest(const Point2D& point, Tour<Point2D>& tour);
**Step 3**. Test your program by clicking on the Run button. This will print the resulting tour, its length and will draw the tour on the screen. 
