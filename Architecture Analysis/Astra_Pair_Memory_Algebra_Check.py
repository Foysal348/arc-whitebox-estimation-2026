from pathlib import Path
import random, math, json
random.seed(7); n=4
H=[[random.gauss(0,1) for j in range(n)] for i in range(n)]
W=[[random.gauss(0,1) for j in range(n)] for i in range(n)]
def tr(A): return list(map(list,zip(*A)))
def mm(A,B): return [[sum(x*y for x,y in zip(row,col)) for col in zip(*B)] for row in A]
def readout(W,H):
    J=[[H[i][j]/(3 if i==j else 1) for j in range(n)] for i in range(n)]
    V=mm(J,W)
    A=mm(tr([[x*x for x in row] for row in W]),V)
    B=mm(tr([[W[i][j]*V[i][j] for j in range(n)] for i in range(n)]),W)
    return [[A[i][j]+2*B[i][j] for j in range(n)] for i in range(n)]
def tensor(a,b,c):
    if a==b==c: return H[a][a]
    if a==b: return H[a][c]
    if a==c: return H[a][b]
    if b==c: return H[b][a]
    return 0.
Z=readout(W,H)
D=[[sum(tensor(a,b,c)*W[a][i]*W[b][i]*W[c][j] for a in range(n) for b in range(n) for c in range(n)) for j in range(n)] for i in range(n)]
N=readout([[-x for x in row] for row in W],H)
err=max(abs(Z[i][j]-D[i][j]) for i in range(n) for j in range(n))
odd=max(abs(Z[i][j]+N[i][j]) for i in range(n) for j in range(n))
cost={}
for d in [0,3,4,5]:
    m=1024//2**d
    f=7**d*(2*m**3-m*m)+sum(18*7**j*(1024//2**(j+1))**2 for j in range(d))
    cost[str(d)]={'one_product_u':f/2**31,'128_products_u':128*f/2**31}
E=1.97e-8; dmean=1e-9; C=194
bound=(math.sqrt(E)+math.sqrt(dmean))**2
out={'scope':'Toy restricted-tensor identity only. No learned model, no ARC accuracy experiment.',
     'n':n,'absolute_identity_error':err,'absolute_odd_parity_error':odd,'strassen_arithmetic':cost,
     'conditional_mean_mse_bound':bound,'conditional_score_ratio_at_floor':bound*.1/(E*C/1024)}
assert err<1e-12 and odd<1e-12
Path(__file__).with_name('algebra_check.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps(out,indent=2))
