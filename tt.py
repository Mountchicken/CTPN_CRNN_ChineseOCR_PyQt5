a=[4,2,3,1,23,11,51,46,231,42,13,8,7,5,3]
sorted_id = sorted(range(len(a)), key=lambda k: a[k], reverse=False)
temp=[]
for i in range(len(a)):
    temp.append(a[sorted_id[i]])
print(temp)