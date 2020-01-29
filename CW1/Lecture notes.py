"""

Just taking down useful knowledge, not making a comprehensive summary

Lecture 1

Binary search of sorted list - O(logN)



Lecture 2

Sorting a list of integers

Insertion sort. O(N^2)

Merging two sorted lists can be done in O(N) (filling in the merged array sequentially)



Lecture 3

Merge sort - keep splitting, merge single elements, come back up. O(NlogN)

Recursion is very inefficient

No need to code my own sorting function. Use np.sort

Back to searching: can we do better than binary search?



Lecture 4

Constant-time search (positive integers): make an array storing the index of an entry at that entry's position ~
Technically O(1) but might not work for other data. Inserting/deleting entries is also O(N)

IP address example; each address is 4 integers. 256^4 possible addresses; not feasible to store a constant-time array
New idea: hash function. Will send an address to an index in a more sensible way

We choose N << 256^4 as our total number of indices (there might be collisions later)

Hash function should: output non-negative integers
                        Send the same input to the same output each time
                        Should distribute evenly amongst indices. I.e. probability of a hash collision is about 1/N

For IP addresses: could try a weighted sum of the integers, mod a prime p. This works
Should choose p roughly equal to N, or maybe 2N in practice to further minimise collisions

Implementation workflow: initialize a dictionary/list where we will store addresses
                            Compute the index for an address
                            Store the address in that location in the table
                            If there's a hash collision, append it to a list in that location instead

Once we have a well-designed hash table, lookup is O(1) (no collisions) or O(sublist lengths), maybe O(1/N)
Insertion and deletion are close to O(1) 'for a careful implementation'

Python provides a hash function that takes in (almost) any input

Can implement a hash table as a list of lists...
But Python dictionaries ARE hash tables. Implement with curly brackets. Example:

key = "123.45.241.12"
value=[14,1,2019,20]
d = {key:value}
d
    Out[9]: {'123.45.241.12': [14, 1, 2019, 20]}
d["123.45.241.12"]
    Out[10]: [14, 1, 2019, 20]

Python applies a hash function to dictionary keys

Initialize a new dictionary: d = dict()
Set a value at key: d[key] = value
Get value at key, else output 1: d.get(key,1)
Check if key is in k: key in d
Length: len(d)
Remove key and its value: del d[key]

"for key in d:" (i.e. iterating through the dictionary) is O(N)

One strand of DNA (one string of the double helix) is made up of ACGT
Codons consist of three DNA bases
Gene sequencing involves finding a base sequence from DNA and analysing its codons' behaviour



Lecture 5

Analyzing running time: count operations (adding, assigning, comparing). Consider worst case

Can characterize running time with big O notation
C is O(f(N)) if there is an integer a s.t. for sufficiently large N, C <= a.f(n)
            I.e. places an asymptotic upper bound

Can similarly place a lower bound. Prasun calls it Omega(f(N))
It's Theta(f(N)) iff it's O(f(N)) and Omega(f(N))

In this coursework: don't need to worry about such definitions.
Ok to give running time as a sum of functions of N in order of dominance
Say it's big O of the leading-order term (though this doesn't match the definition)

Problem: given a sequence of bases and a (shorter) pattern, find all instances of the pattern in the sequence
            String S of length N, pattern P of length M

Naive approach: go through the string one element at a time searching for the pattern
            Could search one pattern element at a time and move on at the first mis-match
            Worst case O(MN) when there are many near-misses

Binary search (for first element) will require storing N length-M strings/arrays

Hash table is faster but with wasteful memory usage

Partial solution: rolling hash. Apply hash to P, and apply hash to each length-M substring of S. Then check
            For a good hash function, cost is O(M+N)
E.g. for bases, set A, C, G, T as 0, 1, 2, 3, then send an M-long string to an order M polynomial in 4 with coefficients from the string



Lecture 6

But that partial solution has O(M) operations for all N-M iterations
Can we make use of the fact that we're hashing consecutive substrings?
Yes, multiply by 4 and sort out the first and last terms. 4 operations per hash (except the first one)
Nonetheless for large M, fast arithmetic can be a problem (language-dependent and especially for base 26 rather than 4)
Can alleviate this by taking mod a large prime q each time

This is known as Rabin-Karp. Code implementation given here

"""