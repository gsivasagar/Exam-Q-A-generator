# Algorithms and Data Structures

## Searching and Sorting
Binary search finds an item in a sorted array in O(log n) time by repeatedly halving the search range. Linear search checks every element and takes O(n) time but works on unsorted data. Quicksort picks a pivot and partitions the array around it, with an average time of O(n log n) and a worst case of O(n squared). Merge sort always runs in O(n log n) time and is stable, but it needs extra memory proportional to the input. Insertion sort is efficient for small or nearly sorted arrays.

## Graph Algorithms
Breadth-first search explores a graph level by level using a queue and finds the shortest path in an unweighted graph. Depth-first search follows one branch as far as possible using a stack or recursion and is used for cycle detection and topological sorting. Dijkstra's algorithm finds the shortest paths from one source in a graph with non-negative edge weights using a priority queue. The Bellman-Ford algorithm handles negative edge weights and can detect negative cycles. A minimum spanning tree connects all vertices with the least total edge weight, and Kruskal's algorithm builds it by adding the cheapest edges that avoid cycles.

## Data Structures
A hash table maps keys to buckets using a hash function and offers average O(1) lookup, insertion and deletion. Collisions in a hash table are resolved by chaining or open addressing. A stack follows last-in first-out order, while a queue follows first-in first-out order. A binary heap stores a complete binary tree in an array and gives O(log n) insertion and fast access to the smallest element. A balanced binary search tree such as an AVL tree keeps its height logarithmic by rotating nodes after updates.

## Design Techniques
Dynamic programming solves a problem by storing the answers to overlapping subproblems, as in computing Fibonacci numbers with memoization. Greedy algorithms make the locally best choice at each step and are correct only when the problem has the greedy-choice property. Divide and conquer splits a problem into independent parts, solves them and combines the results. Big-O notation describes an upper bound on how running time grows with input size. Amortized analysis averages the cost of operations over a whole sequence, which explains why appending to a dynamic array is cheap.
