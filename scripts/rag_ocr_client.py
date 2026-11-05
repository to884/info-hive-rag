"""OCR の子プロセスを再利用し、UTF-8 の JSON 行で画像の認識を依頼する。"""
from __future__ import annotations

import json
import os
import select
import subprocess
import sys
import time
from pathlib import Path

from rag_config import Config


class OcrClient:
    def __init__(self,config: Config,command: list[str] | None=None,device: str='cpu'):
        self.config=config
        self.command=command
        if device not in {'cpu','cuda'}: raise ValueError('OCR の device は cpu / cuda にしてください。')
        self.device=device
        self.process=None
        self.log=None

    def __enter__(self): return self
    def __exit__(self,*_): self.close()

    def _start(self):
        self.config.data_dir.mkdir(parents=True,exist_ok=True)
        self.log=(self.config.data_dir/'ocr-worker.log').open('a',encoding='utf-8')
        command=self.command or ['uv','run','--no-project','--script',str(Path(__file__).with_name('rag_ocr.py'))]
        env={**os.environ,'PYTHONUTF8':'1','PYTHONIOENCODING':'utf-8','PYTHONDONTWRITEBYTECODE':'1',
             'UV_CACHE_DIR':str(self.config.uv_cache_dir)}
        self.process=subprocess.Popen([*command,'--worker','--threads',str(self.config.threads),'--device',self.device],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,text=True,encoding='utf-8',
            env=env,bufsize=1,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        if os.name=='nt':
            import ctypes
            import msvcrt
            from ctypes import wintypes
            self._handle=wintypes.HANDLE(msvcrt.get_osfhandle(self.process.stdout.fileno()))
            self._peek=ctypes.WinDLL('kernel32',use_last_error=True).PeekNamedPipe
            self._peek.argtypes=[wintypes.HANDLE,ctypes.c_void_p,wintypes.DWORD,
                                ctypes.c_void_p,ctypes.POINTER(wintypes.DWORD),ctypes.c_void_p]
            self._peek.restype=wintypes.BOOL

    def recognize(self,image: Path,timeout: float=1800)->dict:
        if self.process is None: self._start()
        if self.process.poll() is not None:
            raise RuntimeError('OCR ワーカーが終了しました。保存済みのページから再開してください。')
        self.process.stdin.write(json.dumps({'image':str(image.resolve())},ensure_ascii=False)+'\n')
        self.process.stdin.flush()
        deadline=time.monotonic()+timeout
        while time.monotonic()<deadline:
            if os.name=='nt':
                from ctypes import wintypes,byref
                available=wintypes.DWORD()
                ready=self._peek(self._handle,None,0,None,byref(available),None) and available.value>0
            else:
                ready=bool(select.select([self.process.stdout],[],[],.05)[0])
            if ready:
                line=self.process.stdout.readline()
                if not line: break
                try: response=json.loads(line)
                except ValueError as error:
                    raise RuntimeError('OCR ワーカーの応答を読めませんでした。') from error
                if not response.get('ok'):
                    raise RuntimeError('OCR に失敗しました: '+response.get('error','詳細不明'))
                return response
            if self.process.poll() is not None: break
            time.sleep(.02)
        self.close()
        raise RuntimeError('OCR の応答がないか、待機が時間切れになりました。保存済みページから再開してください。')

    def close(self):
        if self.process is not None:
            try:
                self.process.stdin.close()
                self.process.wait(timeout=5)
            except (OSError,subprocess.TimeoutExpired):
                import psutil
                try:
                    parent=psutil.Process(self.process.pid)
                    for child in reversed(parent.children(recursive=True)):
                        try: child.kill()
                        except psutil.NoSuchProcess: pass
                    parent.kill()
                except psutil.NoSuchProcess: pass
                self.process.wait(timeout=5)
            finally:
                self.process.stdout.close()
                self.process=None
        if self.log is not None:
            self.log.close(); self.log=None
