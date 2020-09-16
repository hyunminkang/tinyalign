# cython: language_level=3

from cpython.mem cimport PyMem_Malloc, PyMem_Free
from cpython cimport array
import cython
import array


@cython.wraparound(False)
@cython.boundscheck(False)
def edit_distance(s, t, int maxdiff=-1, int subcost = 1, int gapcost = 1):
    """
    Return the edit distance between the strings s and t.
    The edit distance is the sum of the numbers of insertions, deletions,
    and mismatches that is minimally necessary to transform one string
    into the other.

    If maxdiff is not -1, then a banded alignment is performed. In that case,
    the true edit distance is returned if and only if it is maxdiff or less.
    Otherwise, a value is returned that is guaranteed to be greater than
    maxdiff, but which is not necessarily the true edit distance.
    """
    cdef:
        array.array path_types = array.array('i',[])
        array.array path_lens  = array.array('i',[])
        unsigned int m = len(s)  # index: i
        unsigned int n = len(t)  # index: j
        int e = maxdiff 
        unsigned int i, j, start, stop, c, smallest
        unsigned int prev
        unsigned int prefix_match, suffix_match  # newly added
        bint match
        bytes s_bytes, t_bytes
        char* sv
        char* tv

    # Return early if string lengths are too different
    if ( gapcost * 2 <= subcost ):
        raise ValueError("subcost %d must be smaller than twice gapcost %d" % (subcost,gapcost))

    cdef unsigned int absdiff = m - n if m > n else n - m
    if e != -1 and absdiff > e:
        return absdiff

    s_bytes = s.encode() if isinstance(s, unicode) else s
    t_bytes = t.encode() if isinstance(t, unicode) else t
    sv = s_bytes # pointer to the cursor for s
    tv = t_bytes # pointer to the cursor for t

    # Skip identical prefixes
    prefix_match = 0
    while m > 0 and n > 0 and sv[0] == tv[0]:
        sv += 1
        tv += 1
        m -= 1
        n -= 1
        prefix_match += 1

    # Skip identical suffixes
    suffix_match = 0
    while m > 0 and n > 0 and sv[m-1] == tv[n-1]:
        m -= 1
        n -= 1
        suffix_match += 1

    cdef unsigned int result
    cdef unsigned int* costs = <unsigned int*>PyMem_Malloc((m + 1) * sizeof(unsigned int))
    if not costs:
        raise MemoryError()
    ## paths - variable needed for backtracking. records # of vertical moves needed before a horizontal move
    cdef unsigned int* paths = <unsigned int*>PyMem_Malloc((m + 1) * sizeof(unsigned int))
    if not paths:
        raise MemoryError()

    with nogil:
        for i in range(m + 1):      # initialize cost[i]
            costs[i] = i * gapcost  # ins_s/del_t for i letters
            paths[i] = 0            # optimal backtrack path is all ins_s/del_t
        if e == -1:
            # Regular (unbanded) global alignment
            prev = 0
            for j in range(1, n + 1):   # j is the index for t
                prev = costs[0]         # prev is stored cost 
                costs[0] += gapcost     # make del_s/ins_t initially
                paths[0] += 1
                for i in range(1, m+1): # i is the index for s
                    match = sv[i-1] == tv[j-1] # 0/1 value indicating the match
                    if ( costs[i] < prev - match ): 
                        if ( costs[i] < costs[i-1] ):   # min=costs[i]=del_s/ins_t
                            c = costs[i] + gapcost
                            paths[i] += 1
                        else:                           # min=costs[i-1]=ins_s/del_t
                            c = costs[i-1] + gapcost
                            paths[i] = 0 
                    elif ( costs[i-1] < prev - match ): # min=costs[i-1]=ins_s/del_t
                        c = costs[i-1] + gapcost
                        paths[i] = 0
                    else:                               # min=prev-match=sub
                        c = prev - match + subcost
                        paths[i] = 1
                    # c = 1 + min(
                    #     prev - match,  
                    #     costs[i],      
                    #     costs[i-1],
                    # )
                    prev = costs[i]
                    costs[i] = c
            result = costs[m]
        else:
            # Banded alignment
            smallest = 0
            for j in range(1, n + 1):           # j is the index for s
                stop = min(j + e + 1, m + 1)    # stop indicates where to stop i
                if j <= e:                      # if j < maxdiff 
                    prev = costs[0]             
                    costs[0] += gapcost
                    paths[0] += 1
                    smallest = costs[0]
                    start = 1                   # start is always 1
                else:
                    start = j - e               # start is j-maxdiff
                    prev = costs[start - 1]
                    paths[start-1] += 1
                    smallest = maxdiff + gapcost
                for i in range(start, stop):
                    match = sv[i-1] == tv[j-1]
                    if ( costs[i] < prev - match ): 
                        if ( costs[i] < costs[i-1] ):   # min=costs[i]=del_s/ins_t
                            c = costs[i] + gapcost
                            paths[i] += 1
                        else:                           # min=costs[i-1]=ins_s/del_t
                            c = costs[i-1] + gapcost
                            paths[i] = -1 
                    elif ( costs[i-1] < prev - match ): # min=costs[i-1]=ins_s/del_t
                        c = costs[i-1] + gapcost
                        paths[i] = -1
                    else:                               # min=prev-match=sub
                        c = prev - match + subcost
                        paths[i] = 0
                    # c = 1 + min(
                    #     prev - match,
                    #     costs[i],
                    #     costs[i-1],
                    # )
                    prev = costs[i]
                    costs[i] = c
                    smallest = min(smallest, c)
                if smallest > maxdiff:
                    break
            if smallest > maxdiff:
                result = smallest
            else:
                result = costs[m]
        ## backtracking paths
        
    PyMem_Free(costs)
    return result


def hamming_distance(unicode s, unicode t):
    """
    Compute hamming distance between two strings. If they do not have the
    same length, an IndexError is raised.

    Return the number of differences between the strings.
    """
    cdef Py_ssize_t m = len(s)
    cdef Py_ssize_t n = len(t)
    if m != n:
        raise IndexError("sequences must have the same length")
    cdef Py_ssize_t e = 0
    cdef Py_ssize_t i
    for i in range(m):
        if s[i] != t[i]:
            e += 1
    return e
