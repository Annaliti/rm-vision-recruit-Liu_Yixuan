"""Train a five-class RoboMaster armor digit CNN, export ONNX, benchmark CPU.

Run after setup.sh: .venv/bin/python experiment.py
The source ZIP supplies paired image and text files (objectness, class_id).
Only actual digit labels 1..5 are used; labels 6..9 are robot symbols, not digits.
"""
import argparse, collections, hashlib, json, os, platform, random, subprocess, sys, time, zipfile
from pathlib import Path
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parent
CLASSES = ['1', '2', '3', '4', '5']
SOURCE = 'https://github.com/baiyeweiguang/Number-Classifier-for-RoboMaster'

class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(1,8,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
                                      nn.Conv2d(8,16,3,padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(16*7*7,32), nn.ReLU(), nn.Linear(32,5))
    def forward(self,x): return self.classifier(self.features(x))

class ArmorDataset(Dataset):
    def __init__(self, records, root, augment=False):
        self.records,self.root,self.augment=records,root,augment
        self.images=[]
        for r in records:
            with Image.open(root/r['path']) as im:
                self.images.append(im.convert('L').resize((28,28),Image.Resampling.BILINEAR).copy())
    def __len__(self): return len(self.records)
    def __getitem__(self,i):
        im=self.images[i].copy()
        if self.augment:
            im=im.rotate(random.uniform(-8,8),resample=Image.Resampling.BILINEAR,
                         translate=(random.randint(-2,2),random.randint(-2,2)),fillcolor=0)
            im=ImageEnhance.Contrast(im).enhance(random.uniform(.8,1.2))
            if random.random()<.2: im=im.filter(ImageFilter.GaussianBlur(.5))
        x=np.array(im,dtype=np.float32)/255.
        if self.augment:x=np.clip(x+np.random.normal(0,.015,x.shape).astype(np.float32),0,1)
        return torch.from_numpy(x[None]), self.records[i]['label']

def save_json(path,data):
    path.write_text(json.dumps(data,indent=2,ensure_ascii=False))

def prepare(zip_path, out, seed):
    dest=ROOT/'data'/'extracted'
    if not zipfile.is_zipfile(zip_path):
        raise ValueError('数据下载不是有效ZIP文件；请检查下载链接或Git LFS指针。')
    dest.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            target=(dest/info.filename).resolve()
            if dest.resolve() not in target.parents and target!=dest.resolve():raise ValueError('Unsafe archive path')
            if (info.external_attr>>16)&0o170000==0o120000:raise ValueError('Archive symlink not allowed')
        z.extractall(dest)
    records=[]; seen={}; raw=collections.Counter(); excluded=collections.Counter(); dup=0
    for p in sorted(dest.rglob('*')):
        if p.suffix.lower() not in {'.png','.jpg','.jpeg','.bmp'} or '__MACOSX' in p.parts:continue
        lab=p.with_suffix('.txt')
        if not lab.exists(): excluded['missing_label']+=1;continue
        try:
            v=lab.read_text().strip().split(); obj,cls=int(float(v[0])),int(float(v[1]))
        except (ValueError,IndexError):excluded['invalid_label']+=1;continue
        raw[str(cls)]+=1
        if obj!=1 or cls not in range(1,6):excluded['non_digit_or_negative']+=1;continue
        with Image.open(p) as im: pixels=im.convert('L').resize((28,28),Image.Resampling.BILINEAR).tobytes()
        digest=hashlib.sha256(pixels).hexdigest()
        if digest in seen:
            if seen[digest]!=cls:raise ValueError('Conflicting labels for identical image')
            dup+=1;continue
        seen[digest]=cls
        records.append({'path':str(p.relative_to(dest)), 'label':cls-1, 'digit':str(cls), 'sha256_pixels':digest})
    rng=random.Random(seed); splits={'train':[],'val':[],'test':[]}
    for label in range(5):
        subset=[r for r in records if r['label']==label];rng.shuffle(subset)
        if len(subset)<20: raise ValueError(f'Class {label+1} has only {len(subset)} images; need at least 20')
        n=len(subset); nt=int(.70*n); nv=int(.15*n)
        for name,items in zip(splits,[subset[:nt],subset[nt:nt+nv],subset[nt+nv:]]):splits[name].extend(items)
    # No byte-identical preprocessed image may cross partitions.
    sets=[{r['sha256_pixels'] for r in splits[k]} for k in splits]
    assert all(not sets[i]&sets[j] for i in range(3) for j in range(i))
    for name,items in splits.items():save_json(out/f'{name}_manifest.json',items)
    summary={'source':SOURCE,'source_zip_sha256':hashlib.sha256(zip_path.read_bytes()).hexdigest(),
             'raw_label_counts':dict(raw),'excluded':dict(excluded),'exact_duplicates_removed':dup,
             'classes':CLASSES,'split_seed':seed,'split_method':'stratified image split 70/15/15 after exact pixel deduplication',
             'limitation':'No capture-session metadata. Near-duplicate adjacent frames may remain; this is an in-dataset test, not an independent-scene generalization claim.',
             'counts':{k:dict(collections.Counter(r['digit'] for r in v)) for k,v in splits.items()}}
    save_json(out/'dataset_summary.json',summary)
    return dest,splits,summary

@torch.inference_mode()
def evaluate(model,loader):
    model.eval(); loss=0.; ys=[]; preds=[]; logits=[]
    for x,y in loader:
        z=model(x);loss+=nn.functional.cross_entropy(z,y,reduction='sum').item()
        ys.extend(y.tolist());preds.extend(z.argmax(1).tolist());logits.append(z.numpy())
    cm=np.zeros((5,5),dtype=np.int64)
    for y,p in zip(ys,preds):cm[y,p]+=1
    recalls=np.diag(cm)/np.maximum(1,cm.sum(1)); prec=np.diag(cm)/np.maximum(1,cm.sum(0))
    f1=2*prec*recalls/np.maximum(1e-12,prec+recalls)
    return {'loss':loss/len(ys),'accuracy':float(np.mean(np.array(ys)==preds)), 'macro_f1':float(f1.mean()),
            'recall_per_class':recalls.tolist(),'confusion_matrix':cm.tolist(),'n':len(ys)},np.concatenate(logits)

def export_benchmark(model,test_ds,out,args):
    import onnx, onnxruntime as ort
    model.eval().cpu();torch.set_num_threads(1)
    onnx_path=out/'armor_digits.onnx'
    torch.onnx.export(model,torch.zeros(1,1,28,28),str(onnx_path),input_names=['image'],output_names=['logits'],
                      dynamic_axes={'image':{0:'batch'},'logits':{0:'batch'}},opset_version=17,do_constant_folding=True)
    onnx.checker.check_model(onnx.load(str(onnx_path)))
    opts=ort.SessionOptions();opts.intra_op_num_threads=1;opts.inter_op_num_threads=1
    opts.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL;opts.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session=ort.InferenceSession(str(onnx_path),sess_options=opts,providers=['CPUExecutionProvider'])
    assert session.get_providers()==['CPUExecutionProvider']
    differences=[];matches=0;n=0; ort_preds=[];labels=[]
    with torch.inference_mode():
        for x,y in DataLoader(test_ds,batch_size=128,shuffle=False):
            a=model(x).numpy();b=session.run(None,{'image':x.numpy()})[0]
            np.testing.assert_allclose(a,b,rtol=1e-4,atol=1e-5)
            differences.append(float(np.max(np.abs(a-b))));matches+=int((a.argmax(1)==b.argmax(1)).sum());n+=len(y)
            ort_preds.extend(b.argmax(1).tolist());labels.extend(y.tolist())
    times={};rng=np.random.default_rng(args.seed)
    for bs in [1,32]:
        arrays=[np.ascontiguousarray(np.stack([test_ds[int(i)][0].numpy() for i in rng.integers(len(test_ds),size=bs)])) for _ in range(100)]
        tensors=[torch.from_numpy(x) for x in arrays]
        raw={'pytorch':[],'onnxruntime':[]};rounds=[]
        with torch.inference_mode():
            for i in range(100):model(tensors[i]);session.run(None,{'image':arrays[i]})
            for round_id in range(args.rounds):
                order=['pytorch','onnxruntime'] if round_id%2==0 else ['onnxruntime','pytorch']
                row={'round':round_id+1}
                for engine in order:
                    measured=[]
                    for i in range(args.iterations):
                        index=i%100
                        t=time.perf_counter_ns()
                        if engine=='pytorch':model(tensors[index])
                        else:session.run(None,{'image':arrays[index]})
                        measured.append((time.perf_counter_ns()-t)/1e6)
                    raw[engine].extend(measured);row[engine+'_median_ms']=float(np.median(measured))
                rounds.append(row)
        stats={k:{'mean_ms':float(np.mean(v)),'median_ms':float(np.median(v)),'p95_ms':float(np.percentile(v,95)),
                  'throughput_images_s':float(bs*1000/np.mean(v))} for k,v in raw.items()}
        stats['speedup_median']=stats['pytorch']['median_ms']/stats['onnxruntime']['median_ms']
        stats['rounds']=rounds;times[str(bs)]=stats;save_json(out/f'raw_latency_batch{bs}.json',raw)
    result={'threads':1,'providers':session.get_providers(),'warmup':100,'rounds':args.rounds,'iterations_per_round':args.iterations,
            'precision':'FP32','scope':'Python API invocation to output; preloaded inputs, no disk I/O, preprocessing, model loading, or postprocessing',
            'pytorch_mode':'eager eval inference_mode; no torch.compile','max_absolute_logit_error':max(differences),
            'top1_agreement':matches/n,'onnx_test_accuracy':float(np.mean(np.array(ort_preds)==labels)),
            'onnx_version':onnx.__version__,'onnxruntime_version':ort.__version__,'benchmarks':times}
    save_json(out/'benchmark.json',result);return result

def figures(out,history,test,ds):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(9,3.1))
    for key in ['train_loss','val_loss']:axes[0].plot([r['epoch'] for r in history],[r[key] for r in history],label=key)
    axes[0].set(xlabel='Epoch',ylabel='Cross entropy');axes[0].legend()
    axes[1].plot([r['epoch'] for r in history],[r['val_accuracy']*100 for r in history],color='#006b73')
    axes[1].set(xlabel='Epoch',ylabel='Validation accuracy (%)');fig.tight_layout();fig.savefig(out/'training.png',dpi=180);plt.close(fig)
    cm=np.array(test['confusion_matrix']);fig,ax=plt.subplots(figsize=(4.4,3.5));im=ax.imshow(cm,cmap='Blues')
    for (i,j),v in np.ndenumerate(cm):ax.text(j,i,str(v),ha='center',va='center',color='white' if v>cm.max()/2 else 'black')
    ax.set(xticks=range(5),yticks=range(5),xticklabels=CLASSES,yticklabels=CLASSES,xlabel='Predicted digit',ylabel='True digit');fig.tight_layout();fig.savefig(out/'confusion.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(5,6,figsize=(7,5.5))
    for label in range(5):
        ids=[i for i,r in enumerate(ds.records) if r['label']==label][:6]
        for j,i in enumerate(ids):axes[label,j].imshow(ds[i][0][0],cmap='gray');axes[label,j].set_title(CLASSES[label]);axes[label,j].axis('off')
    fig.tight_layout();fig.savefig(out/'samples.png',dpi=180);plt.close(fig)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--epochs',type=int,default=15);ap.add_argument('--seed',type=int,default=20261020)
    ap.add_argument('--iterations',type=int,default=1000);ap.add_argument('--rounds',type=int,default=5)
    ap.add_argument('--zip',type=Path,default=ROOT/'data/armors.zip');ap.add_argument('--output',type=Path,default=ROOT/'results')
    ap.add_argument('--train-only',action='store_true',help='Do not claim ONNX or speed comparison completion in this mode')
    args=ap.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);torch.set_num_threads(4);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    dest,splits,dataset_summary=prepare(args.zip,out,args.seed)
    ds={k:ArmorDataset(v,dest,augment=k=='train') for k,v in splits.items()}
    loaders={k:DataLoader(v,batch_size=128,shuffle=k=='train',num_workers=0) for k,v in ds.items()}
    model=TinyCNN();optimizer=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.0001)
    counts=np.bincount([r['label'] for r in splits['train']],minlength=5);weights=torch.tensor(counts.sum()/(5*counts),dtype=torch.float32)
    criterion=nn.CrossEntropyLoss(weight=weights);history=[];best=(-1.,float('-inf'));start=time.perf_counter()
    for epoch in range(1,args.epochs+1):
        model.train();total=0.;weight_total=0.
        for x,y in loaders['train']:
            optimizer.zero_grad();loss=criterion(model(x),y);loss.backward();optimizer.step()
            total+=loss.item()*weights[y].sum().item();weight_total+=weights[y].sum().item()
        val,_=evaluate(model,loaders['val']);row={'epoch':epoch,'train_loss':total/weight_total,'val_loss':val['loss'],'val_accuracy':val['accuracy']}
        history.append(row);print(json.dumps(row),flush=True)
        score=(val['accuracy'],-val['loss'])
        if score>best:best=score;torch.save(model.state_dict(),out/'armor_digits.pth');best_epoch=epoch
        save_json(out/'history.json',history)
    training_seconds=time.perf_counter()-start
    model.load_state_dict(torch.load(out/'armor_digits.pth',map_location='cpu',weights_only=True));test,_=evaluate(model,loaders['test'])
    try:chip=subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True,stderr=subprocess.DEVNULL).strip()
    except Exception:chip='Unavailable; architecture '+platform.machine()
    result={'date':time.strftime('%Y-%m-%d %H:%M:%S'),'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,
            'platform':platform.platform(),'architecture':platform.machine(),'cpu':chip,'logical_cpus':os.cpu_count(),
            'training_device':'CPU','training_threads':4,'seed':args.seed,'epochs':args.epochs,'best_epoch':best_epoch,
            'parameters':sum(p.numel() for p in model.parameters()),'training_seconds':training_seconds,
            'test':test,'classes':CLASSES,'class_weights':weights.tolist(),'dataset':dataset_summary}
    save_json(out/'metrics.json',result);figures(out,history,test,ds['test'])
    if not args.train_only:result['benchmark']=export_benchmark(model,ds['test'],out,args);save_json(out/'metrics.json',result)
    print('Finished:',out,flush=True)

if __name__=='__main__':main()
