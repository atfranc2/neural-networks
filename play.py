import itertools
    
perms = itertools.permutations([1,-2,2,2,3,3,4])

for x,y,z,a,b,c,d in perms:
    a2 = x + y + z
    a5 = x + a + b + c
    a3 = x + a + d
    if a2 == 2 and a5 == 5 and a3 == 3:
        print(f"[x={x},y={y},z={z},a={a},b={b},c={c},d={d},]")