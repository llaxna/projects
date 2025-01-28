![Screen Shot 2021-11-19 at 11 56 53 PM](https://github.com/user-attachments/assets/e05f995a-d3cc-40c0-a7f7-8928531559dd)

# The Traveling Salesman Problem (TSP)
Given a list of places to visit and the distances between each pair of places, what is the shortest possible route that a salesman can take to visit each place exactly once and return to the starting place?

This is one of the most famous problems in computer science. However, its importance does not arise from the importance of salesmen taking shortest tours, but rather from the far-reaching applications this problem has in many domains. 

## Difficulty of the Problem
The traveling salesman problem is known to be a difficult problem to solve. A brute-force algorithm requires checking 
*n!*
n! permutations, where *n* is the number of places to visit. No one has until now found a general solution that can solve every instance of the TSP problem in polynomial time.  

In fact, the TSP belongs to an important class of problems that no one has a polynomial time solution for. Finding a polynomial time solution for any of these problems automatically leads to finding a polynomial time solution for all the others. Therefore, the Clay Mathematics Institute offers a 
$1,000,000 prize for finding a poly-time solution for this problem!

## Your Task
You will implement a linked data structure that allows representing tours and performing operations on them. 
