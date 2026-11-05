"""索引の作成・更新の意図を保存し、完了済みのネイティブ処理を再実行しない。"""
from __future__ import annotations
import json
import math
import time
from datetime import timedelta
from pathlib import Path
from rag_config import Config,write_json


def version(table):
    value=table.version
    return value() if callable(value) else value


def statistics(table,name):
    stats=table.index_stats(name)
    if stats is None: return None
    return {key:getattr(stats,key,None) for key in ('num_indexed_rows','num_unindexed_rows','index_type','distance_type','num_indices')}


class Indexes:
    def __init__(self,config: Config,folder: Path,table,progress=None):
        self.config,self.table,self.progress=config,table,progress
        self.path=folder/'index-state.json'
        self.state=json.loads(self.path.read_text(encoding='utf-8')) if self.path.exists() else {'strategy':'ivf-sq-v1','trained_rows':0,'deleted_since_training':0,'rows':0,'history':[]}

    def healthy(self,names,rows):
        present={index.name for index in self.table.list_indices()}
        for name in names:
            if name not in present: return False
            stats=statistics(self.table,name)
            if stats is None or stats['num_unindexed_rows']!=0 or stats['num_indexed_rows']!=rows: return False
        return True

    def record(self,pending,recovered):
        self.state.update(pending['updates'])
        self.state['history']=(self.state.get('history',[])+[{'kind':pending['kind'],'rows':pending['rows'],
            'version_before':pending['version_before'],'version_after':version(self.table),'recovered':recovered,'at':time.time()}])[-20:]
        self.state.pop('pending',None)
        write_json(self.path,self.state)

    def operation(self,kind,names,callback,updates=None):
        pending={'kind':kind,'names':names,'rows':self.table.count_rows(),'version_before':version(self.table),'updates':updates or {}}
        self.state['pending']=pending; write_json(self.path,self.state)
        if self.progress: self.progress(kind,0,1)
        callback()
        if not self.healthy(names,pending['rows']): raise RuntimeError('索引作成後に未索引の行が残っています。保存済みの処理から再試行してください。')
        self.record(pending,False)
        if self.progress: self.progress(kind,1,1)

    def maintain(self):
        table=self.table; rows=table.count_rows()
        if table.schema.field('vector').type.list_size!=self.config.dimension:
            raise ValueError('索引のベクトル次元が設定と異なります。別の保存先で再構築してください。')
        pending=self.state.get('pending')
        if pending and pending['rows']==rows and version(table)>pending['version_before'] and self.healthy(pending['names'],rows):
            self.record(pending,True)
        elif pending:
            # ネイティブ処理が未コミットなら、実際の索引状態から必要な操作を選び直す。
            self.state.pop('pending',None)
        if not rows: return self.state
        self.state['deleted_since_training']+=max(0,self.state.get('rows',0)-rows)
        current={index.name:index for index in table.list_indices()}
        if 'terms_idx' not in current:
            self.operation('全文索引の作成',['terms_idx'],lambda:table.create_fts_index('terms',base_tokenizer='whitespace',stem=False,remove_stop_words=False,ascii_folding=False))
        if 'generation_idx' not in current:
            self.operation('世代索引の作成',['generation_idx'],lambda:table.create_scalar_index('generation',index_type='BITMAP',replace=False))
        names=sorted({index.name for index in table.list_indices()})
        if not self.healthy(names,rows):
            self.operation('索引の更新',names,lambda:table.optimize(cleanup_older_than=timedelta(days=self.config.index_cleanup_age_days),delete_unverified=False))
        current={index.name:index for index in table.list_indices()}
        vector=current.get('vector_idx')
        trained=self.state.get('trained_rows',0)
        wrong_type=vector and vector.index_type.lower().replace('_','')!='ivfsq'
        grew=trained and rows>=trained*self.config.ann_rebuild_growth
        shrank=trained and rows*self.config.ann_rebuild_growth<=trained
        many_deleted=trained and self.state['deleted_since_training']>=max(self.config.ann_min_rows,trained//4)
        if rows>=self.config.ann_min_rows and (vector is None or wrong_type or grew or shrank or many_deleted):
            partitions=max(1,min(4096,math.ceil(math.sqrt(rows))))
            self.operation('ANN の作成' if vector is None else 'ANN の再構築',['vector_idx'],
                lambda:table.create_index(vector_column_name='vector',metric='cosine',index_type='IVF_SQ',num_partitions=partitions,name='vector_idx',replace=True),
                {'trained_rows':rows,'num_partitions':partitions,'deleted_since_training':0,'training_metadata':'created_by_rag'})
        elif vector is not None and not trained:
            # 既存の試作索引には学習行数の公開 API がない。再学習の判定基準だけを現在行数から始める。
            self.state.update(trained_rows=rows,num_partitions=None,training_metadata='baseline_rows_inferred')
        names=sorted({index.name for index in table.list_indices()})
        self.state.update(rows=rows,table_version=version(table),dimension=self.config.dimension,
            indexes={name:statistics(table,name) for name in names},ann_ready='vector_idx' in names)
        write_json(self.path,self.state)
        return self.state
