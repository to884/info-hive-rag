# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["ndlocr-lite @ git+https://github.com/ndl-lab/ndlocr-lite@e57f7c6bd9cfb3de5f12f008e71ad2122bd44bfd"]
# ///
"""NDLOCR-Lite を初回の OCR 時に取得して CPU で実行する。"""

import os
import sys

os.environ["PYTHONUTF8"] = "1"
for stream in (sys.stdin, sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8")

def serve(threads: int,device: str='cpu'):
    """固定版の OCR モデルを一度だけ初期化し、画像ごとの要求を処理する。"""
    import contextlib
    import json
    from pathlib import Path
    from types import SimpleNamespace
    import threading
    import numpy as np
    import onnxruntime as ort
    from PIL import Image
    import cv2
    import ocr

    cv2.setNumThreads(1)
    if device=='cuda':
        ort.preload_dlls(directory='')
        if 'CUDAExecutionProvider' not in ort.get_available_providers():
            raise RuntimeError('CUDA の ONNX Runtime がありません。GPU 用の独立環境を準備してください。')

    package=Path(ocr.__file__).resolve().parent
    args=SimpleNamespace(device=device,det_weights=str(package/'model/deim-s-1024x1024.onnx'),
        det_classes=str(package/'config/ndl.yaml'),det_score_threshold=.2,
        det_conf_threshold=.25,det_iou_threshold=.2,
        rec_weights=str(package/'model/parseq-ndl-24x768-100-tiny-153epoch-tegaki3-r8data-202604.onnx'),
        rec_weights30=str(package/'model/parseq-ndl-24x256-30-tiny-189epoch-tegaki3-r8data-202604.onnx'),
        rec_weights50=str(package/'model/parseq-ndl-24x384-50-tiny-300epoch-tegaki3-r8data-202604.onnx'),
        rec_classes=str(package/'config/NDLmoji.yaml'),enable_tcy=False)
    models=None
    evidence=None
    class BoundedRecognizer:
        def __init__(self,model,gate): self.model,self.gate=model,gate
        def read(self,image):
            with self.gate: return self.model.read(image)
    for line in sys.stdin:
        try:
            request=json.loads(line)
            with contextlib.redirect_stdout(sys.stderr):
                if models is None:
                    detector=ocr.get_detector(args)
                    recognizers=[ocr.get_recognizer(args,args.rec_weights30),ocr.get_recognizer(args,args.rec_weights50),ocr.get_recognizer(args)]
                    profiles=[]
                    for name,model in zip(['detector','rec30','rec50','rec100'],[detector,*recognizers]):
                        # 同じ重みと設定でセッションを再作成し、CPU予算を明示する。
                        options=model.session.get_session_options()
                        options.intra_op_num_threads=threads if name=='detector' else 1
                        options.inter_op_num_threads=1
                        profile_root=os.environ.get('INFO_HIVE_OCR_PROFILE_DIR')
                        options.enable_profiling=device=='cuda' and bool(profile_root)
                        if options.enable_profiling:
                            Path(profile_root).mkdir(parents=True,exist_ok=True)
                            options.profile_file_prefix=str(Path(profile_root)/f'{os.getpid()}-{name}')
                        providers=['CUDAExecutionProvider','CPUExecutionProvider'] if device=='cuda' else ['CPUExecutionProvider']
                        model.session=ort.InferenceSession(model.model_path,options,providers=providers)
                        if device=='cuda' and 'CUDAExecutionProvider' not in model.session.get_providers():
                            raise RuntimeError('OCR モデルが CUDA を使えません。CPU への切り替えは行いません。')
                        profiles.append((name,model))
                    if device=='cuda' and profile_root:
                        detector.detect(np.full((512,512,3),255,dtype=np.uint8))
                        for model in recognizers: model.read(np.full((24,100,3),255,dtype=np.uint8))
                        evidence={}
                        for name,model in profiles:
                            path=Path(model.session.end_profiling())
                            events=json.loads(path.read_text(encoding='utf-8'))
                            cuda_events=sum(event.get('args',{}).get('provider')=='CUDAExecutionProvider' for event in events)
                            if not cuda_events: raise RuntimeError(f'{name} の GPU 実行を確認できませんでした。')
                            evidence[name]={'cuda_node_events':cuda_events,'profile':str(path),'providers':model.session.get_providers()}
                    gate=threading.BoundedSemaphore(threads)
                    models=(detector,*[BoundedRecognizer(model,gate) for model in recognizers])
                image=Path(request['image'])
                with Image.open(image) as original:
                    pixels=np.array(original.convert('RGB'))
                result=ocr._run_ocr_on_image_array(*models,image.name,pixels,str(image.parent),save_viz=False)
            answer={'ok':True,'text':result['text'].strip(),'line_count':result['line_count'],
                    'vertical_line_count':result['vertical_line_count'],'pid':os.getpid(),'device':device}
            if evidence:
                answer['gpu_execution']=evidence
                evidence=None
        except Exception as error:
            answer={'ok':False,'error':str(error),'pid':os.getpid()}
        print(json.dumps(answer,ensure_ascii=False),flush=True)


if __name__ == "__main__":
    if '--worker' in sys.argv:
        import argparse
        parser=argparse.ArgumentParser(description='接続中だけ動作する OCR ワーカー。')
        parser.add_argument('--worker',action='store_true')
        parser.add_argument('--threads',type=int,default=4)
        parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
        args=parser.parse_args()
        if not 1<=args.threads<=256: parser.error('threads は 1〜256 にしてください。')
        serve(args.threads,args.device)
    else:
        from ocr import main

        main()
