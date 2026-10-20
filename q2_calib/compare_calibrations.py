"""对固定角点集比较23张、固定5张和错误格长；不据测距误差挑选照片。"""
import json
from pathlib import Path
import cv2
import numpy as np
from measure_a4 import solve

root=Path(__file__).resolve().parent
out=root/'results/calibration_comparison';out.mkdir(parents=True,exist_ok=True)
baseline=json.loads((root/'results/calibration_19p4mm_20261017/calibration.json').read_text())
photos=root/'data/board/main/20261017_134236_837543'
names=[r['image'] for r in baseline['views']]
corners=[]
for name in names:
    gray=cv2.cvtColor(cv2.imread(str(photos/name)),cv2.COLOR_BGR2GRAY)
    ok,c=cv2.findChessboardCornersSB(gray,(9,6),flags=cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not ok:raise RuntimeError('基准照片此次未检测成功，停止以避免暗中改变样本')
    corners.append(c)
obj=np.zeros((54,3),np.float32);obj[:,:2]=np.mgrid[0:9,0:6].T.reshape(-1,2)
a4=json.loads((root/'a4_corners.json').read_text())['images']
models=[]
for label,indexes,scale in [('23_views',list(range(23)),.0194),('5_views',[0,5,10,15,22],.0194),('wrong_square_80_percent',list(range(23)),.01552)]:
    objects=[obj.copy()*scale for _ in indexes]
    rms,K,d,rs,ts=cv2.calibrateCamera(objects,[corners[i] for i in indexes],tuple(baseline['image_size']),None,None)
    measurements=[solve(e['corners'],K,d) for e in a4]
    model={'label':label,'images':[names[i] for i in indexes],'square_mm':scale*1000,'n':len(indexes),
           'calibration_rms_px':rms,'K':K.tolist(),'distortion':d.ravel().tolist(),
           'board_tvecs': [t.ravel().tolist() for t in ts],
           'a4_distances_m':[x['distance_m'] for x in measurements],
           'a4_group_means_m':[float(np.mean([x['distance_m'] for x in measurements[:3]])),float(np.mean([x['distance_m'] for x in measurements[3:]]))]}
    models.append(model)
base,subset,wrong=models
comparison={'models':models,'selection_rule':'基准23张按文件名排序，固定取第1、6、11、16、23张；不按A4结果选样。',
 'wrong_scale_effect':{'ratio':.8,'max_abs_K_difference':float(np.max(np.abs(np.array(wrong['K'])-np.array(base['K'])))),
 'max_abs_A4_distance_difference_mm':float(np.max(np.abs(np.array(wrong['a4_distances_m'])-np.array(base['a4_distances_m'])))*1000),
 'median_board_translation_norm_ratio':float(np.median(np.linalg.norm(wrong['board_tvecs'],axis=1)/np.linalg.norm(base['board_tvecs'],axis=1)))},
 'front_only_experiment':'未采集独立的全正对照片组，未执行；不能用本批有倾斜的照片假冒。',
 'notes':'各组标定RMS只描述各自参与拟合的数据，23张与5张不能仅凭RMS高低判断泛化效果。'}
(out/'comparison.json').write_text(json.dumps(comparison,ensure_ascii=False,indent=2))
print(json.dumps({'models':[{k:v for k,v in m.items() if k not in ['board_tvecs','images','distortion']} for m in models], 'wrong_scale_effect':comparison['wrong_scale_effect']},ensure_ascii=False,indent=2))
