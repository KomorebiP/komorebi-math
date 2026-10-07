from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from komorebi_math import OperatorGrammar, OperatorRealization, PacketOperator, PacketSystem
from komorebi_math.coordinates.analysis_synthesis import AnalysisSynthesisSystem
from komorebi_math.coordinates.numeric import NumericCoordinateTransform
from komorebi_math.nonlinear.correction import HilbertObstructionProjector, LyapunovSchmidtReduction, RightInverseCorrectionProblem
from komorebi_math.nonlinear.derivatives import CallableDerivativeProvider
from komorebi_math.refinement import WeightedTransferEdge, WeightedTransferGraph

checks=[]
def check(name, condition, detail=None):
    checks.append({'name':name,'passed':bool(condition),'detail':detail})
    if not condition:
        raise AssertionError(name)

def ro(name, array):
    check(name, isinstance(array,np.ndarray) and not array.flags.writeable, {'writeable': bool(array.flags.writeable)})

pbuf=np.array([[1.0]],dtype=np.complex128); ps=PacketSystem([pbuf]); pbuf[0,0]=2
check('packet caller mutation isolated', ps.certificate().valid and ps.projectors[0][0,0]==1)
ro('packet internal ndarray readonly',ps.projectors[0])

g=OperatorGrammar(['X'],{'T':('X','X')}); g.relation('idem',[(1,g.word('T','T')),(-1,g.word('T'))])
buf=np.array([[1.0]],dtype=np.complex128); r=OperatorRealization(g,{'X':1},{'T':buf}); buf[0,0]=2
check('realization caller mutation isolated', r.relation_certificate().valid and r.operators['T'][0,0]==1)
ro('realization internal ndarray readonly',r.operators['T'])

bb=np.array([[3.0]],dtype=np.complex128); op=PacketOperator(ps,ps,((bb,),)); bb[0,0]=9
check('packet operator block mutation isolated',op.reconstruct()[0,0]==3 and op.certificate().valid)
ro('packet operator block readonly',op.blocks[0][0])

m=np.array([[1.0]],dtype=np.complex128); h=HilbertObstructionProjector(m,tolerance=1e-12); m[0,0]=0; sol=h.solve([2])
check('Hilbert cached SVD remains tied to owned matrix',h.matrix[0,0]==1 and sol.vanishes and np.allclose(sol.least_norm_correction,[-2]))
ro('Hilbert matrix readonly',h.matrix)

D=np.array([[1.0]],dtype=np.complex128); R=np.array([[1.0]],dtype=np.complex128); y=np.array([0.1],dtype=np.complex128)
prob=RightInverseCorrectionProblem(D,R,lambda x:np.zeros_like(x),y,radius=.2,lipschitz_bound=0,tolerance=1e-12)
D[0,0]=0; R[0,0]=0; y[0]=10
res=prob.solve()
check('right-inverse problem owns matrices and target',prob.hypothesis_certificate().valid and res.certificate.valid and np.allclose(res.solution,[.1]))
ro('right-inverse linear map readonly',prob.linear_map); ro('right-inverse matrix readonly',prob.right_inverse); ro('right-inverse target readonly',prob.target)

provider=CallableDerivativeProvider(lambda x:np.array([x[0]]),{1:lambda x,d:np.array([d[0][0]])},source_dimension=1,target_dimension=1)
base=np.array([0.0],dtype=np.complex128); ls=LyapunovSchmidtReduction(provider,base,tolerance=1e-12); base[0]=7
check('Lyapunov-Schmidt base point owned',np.allclose(ls.base_point,[0]) and ls.decomposition.certificate.valid)
ro('Lyapunov-Schmidt base point readonly',ls.base_point)

diag=np.array([[2.0]],dtype=np.complex128); em=np.array([[.25]],dtype=np.complex128); edge=WeightedTransferEdge(0,0,1,em)
wg=WeightedTransferGraph(ps,diag,[edge]); diag[0,0]=4; em[0,0]=.75
check('weighted graph diagonal cache coherent',wg.diagonal[0,0]==2 and wg.diagonal_inverse[0,0]==.5)
check('weighted graph edge isolated',wg.edges[0].operator[0,0]==.25 and wg.operator[0,0]==2.25)
ro('weighted graph diagonal readonly',wg.diagonal); ro('weighted graph edge readonly',wg.edges[0].operator)

cm=np.array([[2.0]]); ct=NumericCoordinateTransform('x','y',cm); cm[0,0]=8
check('numeric coordinate control still snapshots',ct.matrix[0,0]==2 and not ct.matrix.flags.writeable)
a=np.array([[2.0]]); s=np.array([[.5]]); ass=AnalysisSynthesisSystem(a,s); a[0,0]=8; s[0,0]=8
check('analysis-synthesis control still snapshots',ass.analysis[0,0]==2 and ass.synthesis[0,0]==.5 and ass.certificate().valid)

result={'schema':'komorebi-math.r4-block6-mutation-audit.v1','assertions':len(checks),'passed':all(x['passed'] for x in checks),'checks':checks}
out=Path(__import__('sys').argv[1]) if len(__import__('sys').argv)>1 else Path('block6_mutation_audit.json')
out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps({'assertions':len(checks),'passed':result['passed']}))
