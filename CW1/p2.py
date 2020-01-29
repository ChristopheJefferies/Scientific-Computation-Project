"""M345SC Homework 1, part 2
Christophe Jefferies 01202145
"""

def nsearch(L,P,target):
    """Input:
    L: list containing *N* sub-lists of length M. Each sub-list
    contains M numbers (floats or ints), and the first P elements
    of each sub-list can be assumed to have been sorted in
    ascending order (assume that P<M). L[i][:p] contains the sorted elements in
    the i+1th sub-list of L
    P: The first P elements in each sub-list of L are assumed
    to be sorted in ascending order
    target: the number to be searched for in L

    Output:
    Lout: A list consisting of Q 2-element sub-lists where Q is the number of
    times target occurs in L. Each sub-list should contain 1) the index of
    the sublist of L where target was found and 2) the index within the sublist
    where the target was found. So, Lout = [[0,5],[0,6],[1,3]] indicates
    that the target can be found at L[0][5],L[0][6],L[1][3]. If target
    is not found in L, simply return an empty list (as in the code below)
    """
    
    Lout = []
    M = len(L[0])
    
    for i, row in enumerate(L):
        #First P sorted elements: binary search for one target match, then
        #search either side of it for the rest
        
        #Initial endpoint and midpoint
        start, end = 0, P-1
        
        while start <= end:
            #Find midpoint
            mid = int(0.5*(start+end))
            
            #If target is found, stop
            if row[mid] == target:
                break
            
            #If mid is too large, bring end down
            elif row[mid] > target:
                end = mid-1
            
            #If mid is too small, bring start up
            else:
                start = mid+1
        
        if start <= end: #Happens here iff target is present
            #Search outwards from imid for all occurrences
            first, last = mid, mid
            
            while row[first] == target:
                #If the row starts and ends with target, break
                #(as Python allows negative list indices)
                if first == -1:
                    break
                
                #Otherwise step down by 1
                first -= 1
                
            while row[last] == target:
                #We assume M>P as stated in the coursework, else this can throw errors
                if last == P:
                    break
                
                last += 1
            
            #Add appropriate entries to Lout
            for j in range(first+1, last):
                Lout.append([i, j])
        
        #Linear search to find target in remaining M-P elements
        Lout += [[i, k] for k in range(P, M) if row[k]==target]

    return Lout




def nsearch_time():
    """Analyze the running time of nsearch.
    Add input/output as needed, add a call to this function below to generate
    the figures you are submitting with your codes.
    """
    
    import time
    import matplotlib.pyplot as plt
    from numpy.random import randint
    
    #Function generating an L with the given N, M, P. Integers inside range from 0 to maxint-1
    def makeL(N, M, P, maxint):
        return [sorted([randint(maxint) for i in range(P)]) + [randint(maxint) for i in range(M-P)] for j in range(N)]
    
    
    
    #Plot 1: average time (over 5 runs) for nsearch to run as N varies, for several M
                #Keep P at M/2 to take account of both behaviours (binary and linear search)
                #Set maxint to M/2 so we expect about two target occurences per sublist
    
    Nrange = [1000*(i+1) for i in range(10)] #Values of N to use (x axis)
    Mvals = [10, 100, 1000] #Values of M to use (one line on plot per value)
    repeats = 5 #Will average over this many runs at each N
    
    timelist = [0 for i in range(repeats)] #Will hold the times taken for each iteration (5 repeats)
    y = [[0 for N in Nrange] for M in Mvals] #Will hold all the averaged times
    plotlist = [0 for M in Mvals] #For easy plotting and legend
    
    #For printing progress
    maxN = max(Nrange)
    Mnum = len(Mvals)-1
    
    for i, M in enumerate(Mvals):
        P = int(M/2) #Justified in the discussion
        
        for j, N in enumerate(Nrange):
            print("M", i, "/", Mnum, ", N", N, "/", maxN) #To track progress
            
            for test in range(len(timelist)):
                
                #Generate an input and pick a target
                L = makeL(N, M, P, int(M/2))
                target = randint(M)
                
                #Measure runtime
                start = time.time()
                nsearch(L, P, target)
                end = time.time()
                timelist[test] = end - start
            
            #Add average time to y
            y[i][j] = sum(timelist)/float(len(timelist)) #Float for accurate division
            
        print("Done M = ", M) #To track progress
    
    #Plotting
    plt.figure()
    for line in range(len(Mvals)):
        plotlist[line], = plt.plot(Nrange, y[line], label = 'M = ' + str(Mvals[line]))
    plt.legend(handles=plotlist)
    plt.title("nsearch runtime for different N and M, with P = M/2 \n Plot by nsearch_time \n Christophe Jefferies")
    plt.xlabel("N")
    plt.ylabel("Average time over 5 runs (seconds)")
    plt.savefig("fig1.png", bbox_inches='tight')
    print("fi1.png saved")
    
    
    
    """
    Plot 2: time taken as M increases for fixed large N
    """
    
    N = 1000
    Mrange = [1000*(i+1) for i in range(10)] #range of N to use
    fracs = [0.1, 0.5, 0.9] #Values of P/M to use. One line will be created per value
    
    y = [[0 for M in Mrange] for frac in fracs] #Will hold all the averaged times
    plotlist = [0 for frac in fracs] #For easy plotting and legend
    
    #For printing progress
    maxM = max(Mrange)
    fracnum = len(fracs)-1
    
    for i, frac in enumerate(fracs):
        
        for j, M in enumerate(Mrange):
            print("frac", i, "/", fracnum, ", M", M, "/", maxM) #To track progess
            
            for test in range(len(timelist)):
                
                #Generate an input and pick a target
                L = makeL(N, M, int(frac*M), M)
                target = randint(M)
                
                #Measure runtime
                start = time.time()
                nsearch(L, int(frac*M), target)
                end = time.time()
                timelist[test] = end - start
            
            #Add average time to y
            y[i][j] = sum(timelist)/float(len(timelist))
        
        #To track progess
        print("Done frac = ", frac)
    
    #Plotting
    plt.figure()
    for line in range(len(fracs)):
        plotlist[line], = plt.plot(Mrange, y[line], label = 'P/M = ' + str(fracs[line]))
    plt.legend(handles=plotlist)
    plt.title("nsearch runtime for different M and ratios P/M, with N = 1000 \n Plot by nsearch_time \n Christophe Jefferies")
    plt.xlabel("M")
    plt.ylabel("Average time over 5 runs (seconds)")
    plt.savefig("fig2.png", bbox_inches='tight')
    
    
    
    """
    Discussion: (add your discussion here)
    
    Algorithm description:
        The algorithm tackles each sublist of L separately.
        For the first P elements of each sublist, we do a binary search to find
        one instance of the target. We then search outwards either side of this
        point for any more instances (making use of the fact that the first P
        elements are in ascending order).
        The algorithm then searches linearly through the remaining M-P elements
        for any more appearances of the target.
    
    Runtime order analysis:
        We are given that N, P, and M-P are all large, so we can ignore the extreme
        worst case where P is close to zero and we are only linearly searching.
        Equally we can ignore the case P close to M, where we are essentially
        doing a binary search.
    
        The binary search of the first P elements is O(logP), as seen in lectures.
        
        The linear search either side of the target instance (if one was found) is
        usually negligible compared to the binary search; an exceptional case
        is when a large fraction of the first P elements are equal to the target,
        resulting in a very short binary search and a O(P) linear search. However,
        it is reasonable to assume that the target usually appears infrequently
        enough that this doesn't affect the leading-term runtime order.
        
        The linear search in the remaining M-P elements is then O(M-P).
    
        We do all of the above N times, so the algorithm's leading-order
        runtime is O(N(logP + M-P))
    
    Efficiency analysis:
        It is possible to vectorise a binary search of N lists of length P, being
        careful to reduce all the lists to the same length at each step. However, I
        found that this was slower for this particular task, because it will take as
        many steps as the worst-case list; since we are looking for any one occurence
        of the target in each list, in general the binary search takes far less than
        logP steps. So tackling each list individually reduces steps taken, by
        enough that vectorising is not worth it.
        
        The small linear search either side of a target occurence could also be
        replaced by larger jumps if we make assumptions about the frequency of
        the target, but this would only affect the coefficient in front of its
        runtime contribution, and we assume this is negligible anyway.
        
        The linear search of the remaining M-P unsorted elements is as good as
        it can be (in terms of leading-order runtime); any approach which sorts
        the list first will require at least M-P considerations, and for an
        unsorted list there is no better approach than checking the elements one by one.
    
    Figures:
        fig1.png shows nsearch's runtime as N varies, for several values of M.
        At each N, the average time over 5 runs is found, to minimise variance
        (as any one run can be lucky, for example finding the target in the first P
        elements on the first try). We can see that for a fixed M and P, the
        dependence on N is roughly linear (one outlier on the green line); this
        both matches the leading-order analysis above, and makes sense because
        we are just doing the same process N times in this non-vectorised approach.
        
        As M increases, the slope of the line increases; this is again to be
        expected from the runtime order analysis, which depends positively on M.
        To explore this for even larger values of M displays similar behaviour
        (but with graphs taking too long to generate on my old laptop).
                
        P was set to M/2 for each M in this plot so that both binary and linear
        searching is taken in to account; considerations of different P values
        (relative to M) are below.
        
        fig2.png shows nsearch's runtime as M varies, this time also varying
        the fraction P/M (i.e. the relative sizes of sorted and unsorted parts
        of M). We fix N=1000 throughout, as dependence on N has been discussed
        above. As expected, increasing M leads to a roughly linear increase in
        runtime, as the M-P part is dominant.
        
        As for P/M, a low fraction means most of the time is spent
        linearly searching through almost M elements, which is slow; if P/M is
        close to 1, most of the work is done by the binary search for any one
        instance of the target, which is efficient. Hence having a partially
        sorted list greatly improves the algorithm's performance.
        
        Under the assumption that P and M-P are large (as stated in the
        coursework), we would expect the linear search of M-P to most heavily
        affect the runtime; this again matches the big O analysis, in which
        dependence on P (for a fixed N) was log-like, whilst dependence on
        M-P was linear.
        
        As a whole, this approach is reasonably efficient, taking a second or
        less for the cases considered here.
    """

if __name__ == '__main__':

    #add call(s) to nsearch here which generate the figure(s) you are submitting
    nsearch_time() #no arguments needed