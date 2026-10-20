"""用已核对的 A4 四角与相机标定参数计算距离；参考距离只用于事后比较。"""
import argparse
import json
import os
from pathlib import Path
import tempfile
import cv2
import numpy as np


def solve(corners, K, distortion, width=.297, height=.210):
    obj=np.array([[-width/2,-height/2,0],[width/2,-height/2,0],
                  [width/2,height/2,0],[-width/2,height/2,0]],np.float64)
    points=np.array(corners,np.float64).reshape(4,2)
    ok,rvec,tvec=cv2.solvePnP(obj,points,K,distortion,flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok: raise ValueError('PnP 失败')
    R,_=cv2.Rodrigues(rvec)
    if np.any((R@obj.T+tvec)[2] <= 0):raise ValueError('PnP 得到相机后方的点')
    projected,_=cv2.projectPoints(obj,rvec,tvec,K,distortion)
    rms=float(np.sqrt(np.mean(np.sum((projected.reshape(-1,2)-points)**2,axis=1))))
    return {'distance_m':float(np.linalg.norm(tvec)),'z_m':float(tvec[2,0]),
            'rvec':rvec.ravel().tolist(),'tvec_m':tvec.ravel().tolist(),'reprojection_rms_px':rms}


def main():
    root=Path(__file__).resolve().parent
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--annotations',type=Path,default=root/'a4_corners.json')
    p.add_argument('--calibration',type=Path,default=root/'results/calibration_19p4mm_20261017/calibration.json')
    p.add_argument('--out',type=Path,default=root/'results/a4_measurement')
    a=p.parse_args()
    source=json.loads(a.annotations.read_text());cal=json.loads(a.calibration.read_text())
    K=np.array(cal['camera_matrix'],np.float64);dist=np.array(cal['distortion_coefficients'],np.float64)
    a.out.mkdir(parents=True,exist_ok=True);(a.out/'annotated').mkdir(exist_ok=True)
    rows=[]
    rng=np.random.default_rng(20261020)
    for entry in source['images']:
        path=root/entry['image'];im=cv2.imread(str(path))
        if im is None:raise ValueError(f'图片不可读 {path}')
        if [im.shape[1],im.shape[0]]!=cal['image_size']:raise ValueError('图像分辨率与标定不一致')
        metadata=json.loads(path.with_suffix('.json').read_text())
        s=solve(entry['corners'],K,dist,source['width_m'],source['height_m'])
        ref=metadata.get('distance_m_measured')
        row={**entry,**s,'reference_distance_m':ref,
             'signed_error_mm':(s['distance_m']-ref)*1000 if ref is not None else None,
             'relative_error_percent':(s['distance_m']-ref)/ref*100 if ref else None}
        # 只分析像素标注的敏感性，不代表实际测距置信区间。
        perturbed=[solve(np.array(entry['corners'])+rng.uniform(-2,2,(4,2)),K,dist)['distance_m'] for _ in range(300)]
        row['pixel_perturbation_2px_distance_sd_mm']=float(np.std(perturbed,ddof=1)*1000)
        row['pixel_perturbation_assumption']='每个坐标独立均匀扰动±2像素，300次；仅敏感性分析'
        rows.append(row)
        corners=np.array(entry['corners'],np.int32)
        cv2.polylines(im,[corners],True,(0,200,0),2)
        for i,point in enumerate(corners):
            cv2.circle(im,tuple(point),5,(0,0,255),-1)
            cv2.putText(im,str(i+1),tuple(point+[8,-8]),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,0,255),2)
        cv2.rectangle(im,(0,0),(im.shape[1],62),(255,255,255),-1)
        cv2.putText(im,f"{entry['id']} | PnP: {s['distance_m']:.4f} m | reference: {ref:.3f} m",(15,25),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,0,0),2)
        cv2.putText(im,f"RMS: {s['reprojection_rms_px']:.3f} px | white paper only",(15,51),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,0,0),1)
        cv2.imwrite(str(a.out/'annotated'/f"{entry['id']}.jpg"),im)
    groups=[]
    for ref in sorted(set(r['reference_distance_m'] for r in rows)):
        subset=[r for r in rows if r['reference_distance_m']==ref];d=np.array([r['distance_m'] for r in subset]);e=d-ref
        groups.append({'reference_m':ref,'n':len(d),'mean_distance_m':float(d.mean()),'sample_sd_mm':float(d.std(ddof=1)*1000),
                       'bias_mm':float(e.mean()*1000),'mae_mm':float(np.abs(e).mean()*1000),'rmse_mm':float(np.sqrt(np.mean(e**2))*1000),
                       'mean_relative_error_percent':float(e.mean()/ref*100)})
    result={'calibration':str(a.calibration.relative_to(root)),'target_size_m':[.297,.210],
            'method':'cv2.solvePnP SOLVEPNP_ITERATIVE; distance=norm(tvec); original pixels + original K + distortion',
            'reference_method':'用户用A4纸边长分段建立的名义参考距离；不是卷尺真值',
            'corner_method':source['method'],'rows':rows,'groups':groups}
    (a.out/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'q2-final-mpl'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,3.8))
    ids=[r['id'] for r in rows];d=[r['distance_m'] for r in rows];refs=[r['reference_distance_m'] for r in rows]
    axes[0].plot(ids,d,'o-',color='#186c85',label='PnP distance');axes[0].plot(ids,refs,'s--',color='#b75a32',label='Paper-based reference')
    axes[0].set(xlabel='Image',ylabel='Distance (m)',title='Six images at two reference distances');axes[0].legend(fontsize=8)
    axes[1].bar(ids,[r['signed_error_mm'] for r in rows],color='#186c85');axes[1].axhline(0,color='black',lw=.7)
    axes[1].set(xlabel='Image',ylabel='PnP minus reference (mm)',title='Deviation from approximate reference')
    fig.tight_layout();fig.savefig(a.out/'distance_errors.png',dpi=180);plt.close(fig)
    # 无插值曲线：两个距离只能支持这两个位置的观测。
    fig,ax=plt.subplots(figsize=(6,3.3))
    for i,g in enumerate(groups):
        subset=[r for r in rows if r['reference_distance_m']==g['reference_m']]
        ax.scatter([g['reference_m']]*len(subset),[r['signed_error_mm'] for r in subset],color='#186c85',alpha=.7)
        ax.scatter([g['reference_m']],[g['bias_mm']],marker='_',s=500,color='#b75a32',label='Group mean' if i==0 else None)
    ax.axhline(0,color='black',lw=.7);ax.set(xlabel='Paper-based reference distance (m)',ylabel='Deviation (mm)',title='Two sampled distances only');ax.legend();fig.tight_layout();fig.savefig(a.out/'error_vs_distance.png',dpi=180);plt.close(fig)
    montage=np.vstack([np.hstack([cv2.resize(cv2.imread(str(a.out/'annotated'/f'A{i}.jpg')),(640,360)) for i in [1,2,3]]),
                       np.hstack([cv2.resize(cv2.imread(str(a.out/'annotated'/f'A{i}.jpg')),(640,360)) for i in [4,5,6]])])
    cv2.imwrite(str(a.out/'all_annotated.jpg'),montage)
    print(json.dumps({'rows':[{k:r[k] for k in ['id','distance_m','z_m','signed_error_mm','reprojection_rms_px']} for r in rows],'groups':groups},ensure_ascii=False,indent=2))


if __name__=='__main__': main()
