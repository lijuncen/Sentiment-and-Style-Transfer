# -*- coding: gb18030 -*-   

def variousen_strings(strings, n):
    splitted_strings = [s[0].split(' ') for s in strings]
    vss = vaiousen_splitted_strings(splitted_strings, n)
    return vss

def vaiousen_splitted_strings(splitted_strings, n):
    if n > len(splitted_strings):
        return range(len(splitted_strings))
    
    candidates = list()
    candidates_index = list()
    candidates.append(splitted_strings[0])
    candidates_index.append(0)
    
    search_scope = range(1, len(splitted_strings))
    scores = [0] * (len(splitted_strings))
    
    last_candidate = splitted_strings[0]
    
    while len(search_scope) != 0 and len(candidates) < n:
        max_score = 0 
        max_index = 0
        
        for i in xrange(0, len(search_scope)):
            update_score = levenshtein(splitted_strings[search_scope[i]], last_candidate)
            scores[search_scope[i]] += update_score
            if max_score < scores[search_scope[i]]:
                max_index = i
                max_score = scores[search_scope[i]]
    
        last_candidate = splitted_strings[search_scope[max_index]]
        candidates.append(splitted_strings[search_scope[max_index]])
        candidates_index.append(search_scope[max_index])
        del search_scope[max_index]
    
    return candidates_index
    
''' levenshtein distance'''
def levenshtein(first, second):  
    if len(first) > len(second):  
        first, second = second, first  
    if len(first) == 0:  
        return len(second)  
    if len(second) == 0:  
        return len(first)  
    first_length = len(first) + 1  
    second_length = len(second) + 1  
    distance_matrix = [range(second_length) for _ in range(first_length)]   
    # print distance_matrix  
    for i in range(1, first_length):  
        for j in range(1, second_length):  
            deletion = distance_matrix[i - 1][j] + 1  
            insertion = distance_matrix[i][j - 1] + 1  
            substitution = distance_matrix[i - 1][j - 1]  
            if first[i - 1] != second[j - 1]:  
                substitution += 1  
            distance_matrix[i][j] = min(insertion, deletion, substitution)  
    return distance_matrix[first_length - 1][second_length - 1]  

if __name__ == '__main__':
    ss = ['how are you', 'how are', 'where are you from', 'hello']
    ss_v = variousen_strings(ss, 3)
    for s in ss_v:
        print s
    
