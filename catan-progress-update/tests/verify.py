"""Headless regression checks against the supplied originals."""
import ast
import importlib.util
import sys
import types
from pathlib import Path
from dataclasses import dataclass
from copy import deepcopy
import unittest
import numpy as np
from scipy import sparse

ROOT=Path(__file__).parent
def module(name, path=None):
    if path is None:
        m=types.ModuleType(name);m.__path__=[];sys.modules[name]=m;return m
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

for name in ['catan','catan.core','catan.gui','catan.gui.data','catan.gui.background_tasks','catan.gui.structures','catan.gui.panels','catan.gui.panels.helper']:
    module(name)
changes=module('catan.core.changes')
changes.ChangeKind=type('ChangeKind',(),{})
changes.DataChange=type('DataChange',(),{})
runtime=module('catan.gui.background_tasks.runtime',ROOT/'updated/runtime.py')

def package(prefix, updated):
    m=module(prefix)
    for name in ['dimensions','queries','parameters','types','sparse_values','table','blockwise','requested_rows','distance_statistics','overlap_statistics']:
        path=ROOT/('updated' if updated else 'originals')/(name+'.py')
        if not path.exists():path=ROOT/(name+'.py')
        setattr(m,name,module(prefix+'.'+name,path))
    return m
old=package('oldstats',False);new=package('newstats',True)

@dataclass(frozen=True)
class NeuronComponent:
    neuron_id: int
    session_id: int|None=None
module('catan.gui.structures.state').NeuronComponent=NeuronComponent
actions=module('catan.gui.data.curation_actions');actions.ManipulationSpec=type('ManipulationSpec',(),{})
threshold=module('catan.gui.panels.helper.Threshold');threshold.ThresholdAxis=str
# Use the real non-UI threshold dataclass from the upload.
source=(ROOT/'Threshold.py').read_text();tree=ast.parse(source)
node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ThresholdSpec')
exec('from dataclasses import dataclass\nfrom typing import Literal, Optional\nimport numpy as np\nThresholdDirection=Literal["greater", "less"]\n'+ '\n'.join(source.splitlines()[node.decorator_list[0].lineno-1:node.end_lineno]),threshold.__dict__)
threshold.ThresholdSpec.validate=lambda self:None
api=module('catan.gui.data.statistics');api.PickTable=new.table.PickTable;api.StatisticQuery=new.queries.StatisticQuery
module('catan.gui.data.statistics.engine').StatisticEngine=type('StatisticEngine',(),{})
oldcur=module('oldcur',ROOT/'originals/curation_filter.py');newcur=module('newcur',ROOT/'updated/curation_filter.py')
module('PySide6')
qt=module('PySide6.QtCore');qt.QObject=object;qt.Qt=types.SimpleNamespace();qt.Signal=lambda *a:None
changes.Change=type('Change',(),{})
new.engine=module('newstats.engine',ROOT/'updated/engine.py')
image=module('catan.core.image_correlation')
image.compact_positive_footprint=lambda f,dims:f.toarray().reshape(dims)
image.footprint_similarity_peak=lambda a,b,**kw:float(np.minimum(a,b).sum()/max(1e-12,np.maximum(a,b).sum()))
old.footprint_pairs=module('oldstats.footprint_pairs',ROOT/'originals/footprint_pairs.py')
new.footprint_pairs=module('newstats.footprint_pairs',ROOT/'updated/footprint_pairs.py')

class Checks(unittest.TestCase):
    def run_reported(self, fn):
        values=[];messages=[];clock=[1.0]
        def tick():clock[0]+=.51;return clock[0]
        runtime.monotonic=tick
        ctx=runtime.TaskContext(lambda:False,values.append,messages.append)
        token=runtime.bind_task_context(ctx)
        try:result=fn()
        finally:runtime.reset_task_context(token)
        self.assertEqual(values[-1],100)
        self.assertEqual(values,sorted(set(values)))
        self.assertEqual(values.count(100),1)
        self.assertTrue(messages)
        self.assertIsInstance(runtime.current_task_context(),runtime.NullTaskContext)
        return result,values,messages
    def compare(self,a,b):
        np.testing.assert_allclose(a.values,b.values,equal_nan=True)
        self.assertEqual(a.dims,b.dims)
        for name in ['positions','errors_low','errors_high','n']:
            x=getattr(a,name,None);y=getattr(b,name,None)
            if x is None:self.assertIsNone(y)
            else:np.testing.assert_allclose(x,y,equal_nan=True)
    def test_nested_completion_and_restore(self):
        def f():
            with runtime.progress_scope(message='Outer'):
                with runtime.progress_scope(0,50,'First'):
                    runtime.current_task_context().progress(100)
                with runtime.progress_scope(50,100,'Second'):
                    runtime.current_task_context().progress(100)
        _,v,_=self.run_reported(f)
        self.assertTrue(any(0<x<50 for x in v))
        self.assertTrue(any(50<x<100 for x in v))
    def test_cancel_and_failure_never_complete(self):
        for exception in [runtime.TaskCancelled,ValueError]:
            values=[];ctx=runtime.TaskContext(lambda:False,values.append,lambda m:None)
            token=runtime.bind_task_context(ctx)
            try:
                with self.assertRaises(exception):
                    with runtime.progress_scope():
                        with runtime.progress_scope(0,100):
                            runtime.current_task_context().progress(100)
                            raise exception()
                self.assertIs(runtime.current_task_context(),ctx)
                self.assertNotIn(100,values)
            finally:runtime.reset_task_context(token)
    def test_throttle_final_bypass(self):
        runtime.monotonic=lambda:1.0
        values=[];ctx=runtime.TaskContext(lambda:False,values.append,lambda m:None)
        for x in [0,20,50,88,99,100]:ctx.progress(x)
        self.assertEqual(values,[0,100])
    def test_blockwise_numeric(self):
        rng=np.random.default_rng(3);data=rng.random((17,7,5));data[0,0,:]=np.nan
        for method in ['sum','mean','median','min','max']:
            for select in [None,lambda v:v>.45,lambda v:np.zeros(v.shape,bool)]:
                def run(p):
                    dims={d:p.dimensions.Dimension(name="test", coords=np.arange(n)) for d,n in zip(['neuron','session_i','session_j'],data.shape)}
                    q=p.queries.StatisticQuery('test', reductions=(('session_i',p.queries.ReductionSpec(method)),('session_j',p.queries.ReductionSpec('mean'))))
                    return p.blockwise.evaluate_blockwise(dims,lambda coords:data[np.ix_(*(coords[d] for d in dims))],q,block_values=40,select_values=select)
                expected=run(old);actual,_,_=self.run_reported(lambda:run(new));self.compare(expected,actual)
    def test_empty_blockwise(self):
        for shape in [(0,3),(5,0),(0,0)]:
            for select in [None,lambda v:v>=0]:
                def run(p):
                    dims={d:p.dimensions.Dimension(name="test", coords=np.arange(n)) for d,n in zip(['neuron','session'],shape)}
                    q=p.queries.StatisticQuery('test',reductions=(('session',p.queries.ReductionSpec('sum')),))
                    return p.blockwise.evaluate_blockwise(dims,lambda coords:np.empty(tuple(len(coords[d]) for d in dims)),q,select_values=select,block_values=4)
                a=run(old);b,_,_=self.run_reported(lambda:run(new));self.compare(a,b)
    def test_requested_rows(self):
        for refs in [np.arange(50),np.array([],dtype=int),np.full(30,999)]:
            def run(p):
                dims={'neuron':p.dimensions.Dimension(name="test", coords=np.arange(50))}
                return p.requested_rows.evaluate_requested_rows(dims,lambda r:r['neuron']/50,{'neuron':refs},block_rows=7,select_values=lambda v:v>.5)
            a=run(old);b,_,_=self.run_reported(lambda:run(new));self.compare(a,b)
    def test_sparse_reductions_and_tables(self):
        for method in ['mean','median','sum','max']:
            def run(p):
                dims={'neuron':p.dimensions.Dimension(name="test", coords=np.arange(300)),'session':p.dimensions.Dimension(name="test", coords=np.arange(4))}
                a=p.sparse_values.SparseStatisticArray(name='t',title='t',values=np.arange(600.)/600,positions=np.column_stack((np.repeat(np.arange(300),2),np.tile([0,2],300))),dimensions=dims)
                with runtime.progress_scope(message='Reduce and build'):
                    with runtime.progress_scope(0,70):a.reduce_dimension('session',p.queries.ReductionSpec(method))
                    with runtime.progress_scope(70,100):t=p.table.PickTable.from_stat(a)
                return a,t
            a,ta=run(old);(b,tb),_,_=self.run_reported(lambda:run(new));self.compare(a,b);np.testing.assert_allclose(ta.values,tb.values)
    def test_overlap_cartesian_and_requested(self):
        rng=np.random.default_rng(7)
        footprints=[sparse.csc_matrix(rng.random((30,9))*(rng.random((30,9))>.8)) for _ in range(3)]
        sessions=[types.SimpleNamespace(footprints=f,dims=(5,6),included=None,status={'spatial_loaded':True,'aligned':True}) for f in footprints]
        data=types.SimpleNamespace(sessions=sessions,assignments=types.SimpleNamespace(union=None),alignment_is_stale=lambda sid:False)
        state=types.SimpleNamespace(assignments=np.tile(np.arange(9)[:,None],(1,3)))
        for weighted in [False,True]:
            for directional in [False,True]:
                for pairs in [None,np.array([[0,1,0,1],[2,4,1,2],[5,6,0,2]]),np.empty((0,4),int)]:
                    def run(p):
                        return p.overlap_statistics.overlap_query(data,state,p.queries.StatisticQuery('footprint_overlap',reductions=()),weighted=weighted,directional=directional,requested_pairs=pairs)
                    a=run(old);b,_,_=self.run_reported(lambda:run(new));self.compare(a,b)
    def test_curator_nested_and_empty(self):
        # Real curator projection and combination, deterministic in-memory engine.
        for empty in [False,True]:
            for operator in ['and','or']:
                def run(c):
                    class Engine:
                        state=types.SimpleNamespace(assignments=np.zeros((300,2),int))
                        data=types.SimpleNamespace(assignments=types.SimpleNamespace(union=None),sessions=[])
                        def evaluate_matching_table(self,query,select_values,**kwargs):
                            with runtime.progress_scope(message='Test statistic'):
                                a=new.types.StatisticArray(name=query.statistic_key,title='test',values=np.arange(300)/300,dimensions={'neuron':new.dimensions.Dimension(name="test", coords=np.arange(300))})
                                t=new.table.PickTable.from_stat(a)
                                return t.subset_rows(np.flatnonzero(select_values(t.values)))
                    conditions=[c.CurationFilterCondition(new.queries.StatisticQuery(str(i),reductions=()),threshold.ThresholdSpec(value=(2 if empty else i/5),direction='greater',active=True),id=str(i)) for i in range(3)]
                    group=c.CurationFilterGroup(operator=operator,match_level='neuron',children=[conditions[0],c.CurationFilterGroup(match_level='neuron',children=conditions[1:],id='sub')],id='root')
                    return c.CurationFilterEvaluator(Engine()).evaluate(group)
                a=run(oldcur);b,v,_=self.run_reported(lambda:run(newcur));self.assertEqual(a.neurons,b.neurons);self.assertGreater(len(v),5)
    def test_similarity_candidate_progress(self):
        rng=np.random.default_rng(5)
        sessions=[types.SimpleNamespace(dims=(5,5),centroids=rng.random((30,2)),footprints=sparse.csc_matrix(rng.random((25,30)))) for _ in range(3)]
        data=types.SimpleNamespace(sessions=sessions)
        state=types.SimpleNamespace(assignments=np.tile(np.arange(30)[:,None],(1,3)))
        for relation in ['same','different']:
            def run(p):
                return p.footprint_pairs.calculate_footprint_pairs(data,state,filters=(p.queries.PairFilter('neuron',relation),))
            a=run(old);b,v,_=self.run_reported(lambda:run(new));self.compare(a,b);self.assertGreater(len(v),10)
    def test_engine_cache_and_matching_paths(self):
        from threading import RLock
        for query_getter in [False,True]:
            engine=new.engine.StatisticEngine.__new__(new.engine.StatisticEngine)
            engine.data=types.SimpleNamespace(sessions=[object()]);engine.state=object()
            engine._cache_lock=RLock();engine._cache={};engine._revisions={}
            engine.cache_max_entries=8;engine.cache_max_bytes=1000000
            def stat():return new.types.StatisticArray(name='custom',title='Custom',values=np.arange(600.).reshape(300,2),dimensions={'neuron':new.dimensions.Dimension('neuron',coords=np.arange(300)),'session':new.dimensions.Dimension('session',coords=np.arange(2))})
            def getter(data,state,query,select_values=None,**kwargs):
                dims=stat().dimensions
                return new.blockwise.evaluate_blockwise(dims,lambda c:stat().values[np.ix_(c['neuron'],c['session'])],query,select_values=select_values,block_values=50)
            getter.supports_value_selection=True
            definition=types.SimpleNamespace(key='custom',category='neuron',component_axes=None,curation_kind=None,relationship_axes=None,independent_pairs=False,compact_values=True,dims=('neuron','session'),parameters=(),resolve_parameters=lambda p:{},title_with_parameters=lambda p:'Custom',query_getter=getter if query_getter else None,get_values=lambda **kw:stat())
            engine.registry={'custom':definition}
            q=new.queries.StatisticQuery('custom',reductions=(('session',new.queries.ReductionSpec('mean')),))
            for repeat in range(2):
                table,v,msg=self.run_reported(lambda:engine.evaluate_matching_table(q,lambda v:v>100))
                np.testing.assert_allclose(table.values,np.arange(100,600,2)+.5)
                self.assertTrue(any('table' in m.lower() for m in msg))

if __name__=='__main__':unittest.main(verbosity=2)
