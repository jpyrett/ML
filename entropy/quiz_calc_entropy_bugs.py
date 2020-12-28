# quiz:  Which of the following splitting criteria provides the most information gain for discriminating Mobugs from Lobugs?
# from the: ml-bugs 
#   choices:  Color = brown, green, or blue
#             length < 17   (correct answer)  Infor gain =  0.1126
#             length < 20    

def two_group_ent(first, tot):                        
    return -(first/tot*np.log2(first/tot) +           
             (tot-first)/tot*np.log2((tot-first)/tot))

tot_ent = two_group_ent(10, 24)                       
g17_ent = 15/24 * two_group_ent(11,15) +              
           9/24 * two_group_ent(6,9)                  

answer = tot_ent - g17_ent                            