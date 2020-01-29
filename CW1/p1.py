"""M345SC Homework 1, part 1
Christophe Jefferies 01202145
"""

def ksearch(S,k,f,x):
    """
    Search for frequently-occurring k-mers within a DNA sequence
    and find the number of point-x mutations for each frequently
    occurring k-mer.
    Input:
    S: A string consisting of A,C,G, and Ts
    k: The size of k-mer to search for (an integer)
    f: frequency parameter -- the search should identify k-mers which
    occur at least f times in S
    x: the location within each frequently-occurring k-mer where
    point-x mutations will differ.

    Output:
    L1: list containing the strings corresponding to the frequently-occurring k-mers
    L2: list containing the locations of the frequent k-mers
    L3: list containing the number of point-x mutations for each k-mer in L1.
    """

    N = len(S)
    d = dict()
    L1, L2, L3 = [], [], []

    #Fill in the hash table (dictionary)
    for i in range(N-k+1):
        kmer = S[i:i+k] #Extract the k-mer at the ith position
        
        #If it's already appeared, append i to the list of indices
        if d.get(kmer): #this is a bit faster than "if kmer in d"
            d[kmer].append(i)
            
        #If it hasn't appeared, make a new list containing just i
        else:
            d[kmer] = [i]
    
    #Make L1 and L2
    for kmer in d:
        #If a kmer appears at least f times, add it to L1 and its appeareance indices to L2
        if len(d[kmer]) >= f:
            L1.append(kmer)
            L2.append(d[kmer])
    
    #Make L3
    L3 = len(L1)*[0] #Initialise L3 as a list of zeros of the same length as L1 and L2
    bases = set(['A', 'C', 'G', 'T'])
    
    for i, kmer in enumerate(L1):
         lk = list(kmer) #Strings are immutable so work with lists instead
         xbase = lk[x] #The base at position x in the k-mer
         otherbases = bases - set([xbase]) #set of the other three bases
         
         #Count the number of point-x mutations using the dictionary
         for base in otherbases:
             lk[x] = base #RChange the base in position x
             lks = ''.join(lk) #Form string for that point-x mutation
             L3[i] += len(d.get(lks, [])) #Increase the counter. If key not present, increase by 0

    return L1,L2,L3

"""
Algorithm description:
    
    We start by making a hash table in the form of a dictionary.
    The keys are k-long substrings of S, and each entry is a list of indices where that substring appears.
    
    To check if a k-mer appears at least f times, we just check the length of the list at that key.
    If this is the case, we append the k-mer to L1, and its indices to L2.
    
    To make L3: for each k-mer in L1, we form the three strings corresponding to its point-x mutations.
    We then just check the dictionary for these keys, and increase the corresponding entry of L3.

Runtime discussion:
    
    Using a hash table is evidently much faster than any approach based on
    iterating through the string. Once the table is formed, finding indices and
    point-x mutations are both very quick - indexing and checking for keys are
    both approximately O(1) as discussed in the lectures.
    
    To use a rolling hash and Rabin-Karp is also possible. However, I found this
    was often slower (in Python) than the above.
    This might be because Python is an interpreted language, so repeatedly
    interpreting a written-out rolling hash and table becomes slower (for large
    inputs) than implementing a dictionary (which is essentially a front-end
    for quick, compiled C++ code).
    Additionally, Python's built-in hash function is O(1) for strings, so there
    is little runtime advantage in coding out the hash function specifically.
    However, a dictionary does use more memory than some other approaches. This
    isn't too much of a problem unless we are using really enormous strings.
    
    Appending entries to lists is quite slow in Python, but it's unavoidable
    here; we don't know how many distinct k-mers there will be, nor how many
    times each appears, until we've been through the whole of S.
    Preallocating L3 (once L1 and L2 are made) makes a small difference for
    very large inputs.
    
    Leading-order runtime analysis:
        Forming the table: worst case (N-k+1).(k+1+1) operations
            Here I assume extracting a k-long substring is O(k)...
            It might be O(1) but the documentation doesn't make it clear.
        Making L1 and L2: worst case (N-k+1)(1+1+1) operations
            Worst case is when every k-long substring of S is distinct.
        Making L3: worst case (N-k+1)(k + 3(k+1)) operations
            The final k dependence is because converting strings to/from lists
            depends on their length.
            The 3 comes from the loop through a k-mer's three point-x mutations.
        
        Added together and small/constant terms ignored, this all becomes
        O((N-k)k), or more simply O(Nk) (especially if we assume k is much
        smaller than N).
    
    The algorithm is able to handle a "very long" string (length 23 million) for
    varying orders of magnitude of k. Small k are almost instant, k 100-900
    takes a minute or two on my laptop, and k a few thousand takes a few
    minutes. Much larger k (10000, 100000) still run on my laptop run but take
    10 minutes or a little more.
    
    To summarise, this approach is quite efficient, and able to handle a
    variety of inputs without ever losing any accuracy, though it does use
    up some memory.
"""

