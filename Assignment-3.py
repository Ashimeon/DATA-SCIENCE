# Assignment-3
def oddnumbers(a,b):
    for i in range(a,b+1):
        if i%2!=0:
            yield (i)
n1=oddnumbers(1,'k')
print(next(n1))