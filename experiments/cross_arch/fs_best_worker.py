"""FlashSparse head-to-head worker: time FlashSparse on its fastest fp16 path. Per graph, build 16x1, 8x1, and 8x1-balance
block formats (timed separately); per N, time all three kernels and record the BEST (min) as the
FlashSparse number (matching FlashSparse's own eval which reports the best config). Crash-isolated.
Requires the FlashSparse extension (FS_Block_gpu, FS_SpMM) built from its public repository.
Usage: fs_best_worker.py <dataset> <out_csv> <preproc_csv>"""
import csv, json, os, sys, time, zlib
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import ra_real_graph_eval as R
import ra_spmm, FS_Block_gpu, FS_SpMM
NS=[64,128,256,512]; WARMUP,TIMED=50,200; PART=32
def seed_for(n,N): return (zlib.crc32(n.encode())+N)%(2**31)

def main():
    name,out_csv,pre_csv=sys.argv[1],sys.argv[2],sys.argv[3]
    e=next(d for d in json.load(open('fgcs_results/paper_combined_datasets.json'))['datasets'] if d['name']==name)
    mat=R.load_dataset(e)
    if mat is None: print(f'[{name}] SKIP'); return
    M0=mat['M']; rp=mat['rowptr']; ci=mat['colind']; nnz=int(rp[-1].item())
    def padrows(pad):
        Mp=M0+((pad-M0%pad)%pad)
        return Mp, torch.cat([rp.int(),rp[-1:].int().repeat(Mp-M0)]).contiguous(), ci.int().contiguous()
    Mp16,ro16,co16=padrows(16); Mp8,ro8,co8=padrows(8)
    # build formats once, timed
    def t(fn):
        torch.cuda.synchronize(); s=time.time(); r=fn(); torch.cuda.synchronize(); return r,(time.time()-s)*1000
    (f16,tp16)=t(lambda: FS_Block_gpu.preprocess_gpu_fs(ro16,co16,Mp16,nnz,16,8))
    (f8,tp8)  =t(lambda: FS_Block_gpu.preprocess_gpu_fs(ro8,co8,Mp8,nnz,8,8))
    try:
        (f8b,tp8b)=t(lambda: FS_Block_gpu.preprocess_gpu_fs_balance(ro8,co8,Mp8,nnz,8,8,PART))
    except Exception as ex:
        f8b=None; tp8b=-1; print(f'[{name}] balance preproc failed: {ex}')
    print(f'[{name}] M0={M0} nnz={nnz} preproc16={tp16:.1f} preproc8={tp8:.1f} preproc8b={tp8b:.1f} ms',flush=True)
    with open(pre_csv,'a',newline='') as f:
        csv.writer(f).writerow([name,M0,nnz,round(tp16,3),round(tp8,3),round(tp8b,3)])
    rpg=rp.int().cuda(); cig=ci.int().cuda(); v=torch.ones(nnz,device='cuda')
    rows=[]
    for N in NS:
        try:
            g=torch.Generator(device='cuda').manual_seed(seed_for(name,N))
            B=torch.randn(M0,N,device='cuda',generator=g); x=B.half().cpu().contiguous()
            C_ref=ra_spmm.spmm_cusparse(rpg,cig,v,B)
            ms_cus=R.measure_ms(lambda: ra_spmm.spmm_cusparse(rpg,cig,v,B),WARMUP,TIMED)
            cand={}
            # warmup + timed for each config; keep output of each for correctness
            FS_SpMM.forward_fp16_16(*f16[:3],x,Mp16,N,M0,WARMUP)
            o16,m16=FS_SpMM.forward_fp16_16(*f16[:3],x,Mp16,N,M0,TIMED); cand['16x1']=(float(m16.item()),o16)
            FS_SpMM.forward_fp16_test(*f8[:3],x,Mp8,N,M0,WARMUP,4)
            o8,m8=FS_SpMM.forward_fp16_test(*f8[:3],x,Mp8,N,M0,TIMED,4); cand['8x1']=(float(m8.item()),o8)
            if f8b is not None:
                FS_SpMM.forward_fp16_balance(f8b[0],f8b[1],f8b[2],f8b[3],f8b[4],x,Mp8,N,M0,WARMUP)
                o8b,m8b=FS_SpMM.forward_fp16_balance(f8b[0],f8b[1],f8b[2],f8b[3],f8b[4],x,Mp8,N,M0,TIMED)
                cand['8x1_balance']=(float(m8b.item()),o8b)
            best=min(cand,key=lambda k:cand[k][0]); bms,bout=cand[best]
            Cfs=bout.cuda().float()[:M0]
            rel=(torch.linalg.norm(Cfs-C_ref)/(torch.linalg.norm(C_ref)+1e-12)).item()
            ok=rel<1e-2; sp=ms_cus/bms if bms>0 else 0
            allms={k:round(vv[0],5) for k,vv in cand.items()}
            m16v=allms.get('16x1'); m8v=allms.get('8x1'); m8bv=allms.get('8x1_balance')
            rows.append([name,N,M0,nnz,'FLASHSPARSE_BEST',best,round(bms,5),round(ms_cus,5),round(sp,4),
                         round(rel,6),ok,m16v,m8v,m8bv,''])
            print(f'[{name}] N={N} best={best} {bms:.4f}ms (16x1={m16v} 8x1={m8v} 8x1b={m8bv}) cusparse={ms_cus:.4f} sp={sp:.2f}x correct={ok} rel={rel:.1e}',flush=True)
            del B,C_ref,Cfs; torch.cuda.empty_cache()
        except Exception as ex:
            rows.append([name,N,M0,nnz,'FLASHSPARSE_BEST','',-1,-1,0,-1,False,None,None,None,f'{type(ex).__name__}:{str(ex)[:80]}'])
            print(f'[{name}] N={N} ERROR {type(ex).__name__}: {str(ex)[:80]}',flush=True); torch.cuda.empty_cache()
    with open(out_csv,'a',newline='') as f: csv.writer(f).writerows(rows)
    print(f'[{name}] done {len(rows)} rows',flush=True)
if __name__=='__main__': main()
