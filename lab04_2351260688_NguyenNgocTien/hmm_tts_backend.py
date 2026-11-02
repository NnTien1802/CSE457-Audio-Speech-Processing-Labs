"""
Lab 4E - A SMALL DATA-CONDITIONED, LEFT-TO-RIGHT GAUSSIAN HMM PARAMETER DEMO.

This is a pedagogical model, NOT an HTS-trained statistical parametric TTS:
- Units are segmented into 3 ordered states by proportional time, not by EM/Baum-Welch.
- State Gaussian means/variances and duration/transition proxies come from actual WAV features.
- Generation creates acoustic-parameter TRAJECTORIES, NOT an intelligible waveform.
- Delta-feature MLPG is implemented via a regularized quadratic least-squares solve.
Do not present it as a trained text-to-waveform HMM synthesizer.
"""
from pathlib import Path
import re
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

SR = 16000
HOP = 160  # 10 ms
FRAME = 1024
N_STATES = 3
FEATURES = ('log_f0', 'log_rms', 'log_centroid')
FLOOR_STD = {'log_f0': .035, 'log_rms': .14, 'log_centroid': .14}
FLOOR_DSTD = {'log_f0': .025, 'log_rms': .075, 'log_centroid': .075}
STEM = {'mời':'moi','khách':'khach','hàng':'hang','số':'so','không':'khong',
        'một':'mot','hai':'hai','ba':'ba','bốn':'bon','năm':'nam',
        'sáu':'sau','bảy':'bay','tám':'tam','chín':'chin','đến':'den','quầy':'quay'}
DIGIT = {'0':'không','1':'một','2':'hai','3':'ba','4':'bốn','5':'năm',
         '6':'sáu','7':'bảy','8':'tám','9':'chín'}


def normalize_queue_text(text):
    """Case study: queue ID digits are read ONE BY ONE."""
    text = re.sub(r'[^\w\s]', ' ', str(text).lower(), flags=re.UNICODE)
    out = []
    for part in text.split():
        if part.isdecimal():
            out.extend(DIGIT[d] for d in part)
        else:
            out.append(part)
    return out


def read_mono_16k(path):
    x, sr = sf.read(str(path), dtype='float32')
    if sr != SR or x.ndim != 1:
        raise ValueError(f'{path}: WAV phải mono 16000 Hz, nhận sr={sr}, ndim={x.ndim}')
    if len(x) < FRAME or not np.isfinite(x).all():
        raise ValueError(f'{path}: âm thanh ngắn hoặc có NaN/Inf')
    return x


def acoustic_frames(x):
    """Aligned 10 ms frame measurements from a real recorded word-unit."""
    y = x.astype(np.float64)
    rms = librosa.feature.rms(y=y, frame_length=FRAME, hop_length=HOP, center=True)[0]
    cent = librosa.feature.spectral_centroid(
        y=y, sr=SR, n_fft=FRAME, hop_length=HOP, center=True)[0]
    flat = librosa.feature.spectral_flatness(
        y=y, n_fft=FRAME, hop_length=HOP, center=True)[0]
    f0 = librosa.yin(y, sr=SR, fmin=75.0, fmax=350.0,
                    frame_length=FRAME, hop_length=HOP, center=True)
    n = min(len(rms), len(cent), len(flat), len(f0))
    rms,cent,flat,f0 = rms[:n],cent[:n],flat[:n],f0[:n]
    # Mask silence and mostly aperiodic frames. This is a HEURISTIC, not ground-truth voicing.
    voiced = (rms > max(0.003, .14 * float(np.max(rms)))) & (flat < 0.28)
    voiced &= np.isfinite(f0) & (f0 >= 75) & (f0 <= 350)
    median_f0 = float(np.median(f0[voiced])) if np.any(voiced) else 120.0
    log_f0 = np.log(np.where(voiced, f0, median_f0))
    feat = np.column_stack((log_f0, np.log(np.maximum(rms,1e-5)),
                            np.log(np.maximum(cent,40.0))))
    delta = np.zeros_like(feat)
    if len(feat) >= 3:
        delta[1:-1] = (feat[2:] - feat[:-2]) / 2.0
        delta[0] = feat[1] - feat[0]
        delta[-1] = feat[-1] - feat[-2]
    return feat,delta,voiced,{'rms':rms,'centroid':cent,'f0':f0}


def locate_candidates(unit_dir, word, max_candidates=3):
    if word not in STEM:
        raise KeyError(f'Từ ngoài inventory: {word!r}')
    return sorted(Path(unit_dir).glob(f'{STEM[word]}_*.wav'))[:max_candidates]


def train_demo_model(unit_dir, tokens, max_candidates=3, n_states=N_STATES):
    """Estimate 3 sequential Gaussian states/word from observed WAVs (fixed alignment)."""
    root = Path(unit_dir)
    words = list(dict.fromkeys(tokens))
    states = {}
    unit_info = []
    for word in words:
        paths = locate_candidates(root, word, max_candidates)
        if not paths:
            raise FileNotFoundError(f'Thiếu unit {word!r}: {root}/{STEM[word]}_01.wav')
        acc = [dict(static=[], delta=[], voiced=[], length=[]) for _ in range(n_states)]
        for path in paths:
            x = read_mono_16k(path)
            v,d,voiced,raw = acoustic_frames(x)
            indices = np.array_split(np.arange(len(v)), n_states)
            for s,ix in enumerate(indices):
                if len(ix)==0: continue
                acc[s]['static'].append(v[ix])
                acc[s]['delta'].append(d[ix])
                acc[s]['voiced'].append(voiced[ix])
                acc[s]['length'].append(len(ix))
            unit_info.append({'word':word,'candidate_file':path.name,
                              'duration_s':len(x)/SR,'n_frames':len(v),
                              'voiced_fraction_heuristic':float(voiced.mean())})
        for s in range(n_states):
            a = acc[s]
            X = np.concatenate(a['static'],axis=0)
            D = np.concatenate(a['delta'],axis=0)
            V = np.concatenate(a['voiced'],axis=0)
            std = np.maximum(X.std(axis=0, ddof=0),
                             np.array([FLOOR_STD[k] for k in FEATURES]))
            dstd = np.maximum(D.std(axis=0, ddof=0),
                              np.array([FLOOR_DSTD[k] for k in FEATURES]))
            frames = float(np.mean(a['length']))
            # Geometric self-loop mean length ~ 1/(1-p_stay).
            p_next = 1.0/max(1.0,frames)
            states[(word,s)] = dict(word=word, state=s, mean=X.mean(axis=0),
                 variance=std**2, delta_mean=D.mean(axis=0),delta_variance=dstd**2,
                 frames=frames,p_stay=1.0-p_next,p_advance=p_next,
                 voiced_probability=float(V.mean()),n_samples=len(X),n_files=len(paths))
    model = {'states':states, 'n_states':n_states,'sr':SR,'hop':HOP,
             'tokens':tokens, 'features':FEATURES}
    model['unit_info'] = pd.DataFrame(unit_info)
    return model


def mlpg_one_dimension(mu, var, delta_mu, delta_var, delta_weight=1.0):
    """Solve static + delta Gaussian likelihood for a whole acoustic parameter trajectory.

    Delta operator: D c[t] = 0.5 * (c[t+1] - c[t-1]) for interior frames.
    This is a SMALL pedagogical MLPG with diagonal variances, not a full HTS system.
    """
    T = len(mu)
    if T < 3: return np.array(mu,dtype=float)
    w0 = 1.0/np.maximum(var,1e-8)
    wd = delta_weight/np.maximum(delta_var,1e-8)
    # Difference operator built as sparse rows; explicit borders.
    rows = []; cols = []; values = []
    for t in range(T):
        if t==0: pair=[(0,-1),(1,1)]
        elif t==T-1: pair=[(T-2,-1),(T-1,1)]
        else: pair=[(t-1,-.5),(t+1,.5)]
        for col,val in pair:
            rows.append(t);cols.append(col);values.append(val)
    from scipy.sparse import coo_matrix
    D = coo_matrix((values,(rows,cols)),shape=(T,T)).tocsr()
    W0 = diags(w0)
    WD = diags(wd)
    H = W0 + D.T@WD@D
    b = w0*mu + D.T@(wd*delta_mu)
    return np.asarray(spsolve(H,b))


def generate_tracks(model, text, duration_scale=1.0, f0_scale=1.0,
                    apply_delta=True, delta_weight=1.0):
    """Conditional HMM *parameter* generation; returns dataframe, states, no waveform."""
    if duration_scale <= 0 or f0_scale <= 0: raise ValueError('Scale phải >0')
    tokens = normalize_queue_text(text)
    if not tokens: raise ValueError('Câu rỗng')
    missing=[w for w in tokens if (w,0) not in model['states']]
    if missing: raise KeyError(f'Model chưa có các từ: {missing}')
    seq=[]
    state_rows=[]
    t=0
    for token_i,word in enumerate(tokens):
        for s in range(model['n_states']):
            st=model['states'][(word,s)]
            frames=max(3,int(round(st['frames']*duration_scale)))
            state_rows.append({'word_position':token_i+1,'word':word,'state':s+1,
                'frames':frames,'duration_s':frames*HOP/SR,
                'self_transition':st['p_stay'],'advance_transition':st['p_advance'],
                'mean_f0_hz':float(np.exp(st['mean'][0])),
                'mean_rms':float(np.exp(st['mean'][1])),
                'mean_spectral_centroid_hz':float(np.exp(st['mean'][2])),
                'voiced_probability':st['voiced_probability']})
            for _ in range(frames):
                seq.append((token_i,word,s,st))
                t+=1
    means=np.array([st['mean'] for _,_,_,st in seq])
    variances=np.array([st['variance'] for _,_,_,st in seq])
    delta_means=np.array([st['delta_mean'] for _,_,_,st in seq])
    delta_vars=np.array([st['delta_variance'] for _,_,_,st in seq])
    out=means.copy()
    if apply_delta:
        for k in range(len(FEATURES)):
            out[:,k]=mlpg_one_dimension(means[:,k],variances[:,k],
                         delta_means[:,k],delta_vars[:,k],delta_weight)
    vp=np.array([st['voiced_probability'] for _,_,_,st in seq])
    f0=f0_scale*np.exp(out[:,0])
    f0[vp<.35]=np.nan  # unvoiced/silent states should NOT get a fundamental frequency
    df=pd.DataFrame({
        'time_s':np.arange(len(seq))*HOP/SR,
        'word_position':[v[0]+1 for v in seq],
        'word':[v[1] for v in seq],
        'state':[v[2]+1 for v in seq],
        'f0_hz':f0,
        'f0_unmasked_hz':f0_scale*np.exp(out[:,0]),
        'rms_amplitude':np.exp(out[:,1]),
        'spectral_centroid_hz':np.exp(out[:,2]),
        'raw_state_f0_hz':f0_scale*np.exp(means[:,0]),
        'raw_state_rms':np.exp(means[:,1]),
        'raw_state_centroid_hz':np.exp(means[:,2]),
        'voiced_probability':vp,
    })
    states_df=pd.DataFrame(state_rows)
    return df,states_df


def export_transition_table(model):
    rows=[]
    for (word,s),st in model['states'].items():
        rows.append({'word':word,'state':s+1,
                     'transition_self_probability':st['p_stay'],
                     'transition_next_probability':st['p_advance'],
                     'avg_duration_frames':st['frames'],
                     'avg_duration_ms':st['frames']*HOP/SR*1000,
                     'trained_wav_count':st['n_files'],
                     'training_frames':st['n_samples']})
    return pd.DataFrame(rows).sort_values(['word','state']).reset_index(drop=True)


def synthesize_hmm(text, unit_dir='lab4_data/units', duration_scale=1.0,
                   f0_scale=1.0):
    """API analogous to Lab 4 (section 5.5); NO waveform resynthesis.

    Returns (None, SR, tracks_dict). None explicitly indicates no generated WAV.
    """
    tokens=normalize_queue_text(text)
    model=train_demo_model(unit_dir,tokens)
    tracks,states=generate_tracks(model,text,duration_scale=duration_scale,f0_scale=f0_scale)
    return None,SR,{'frames':tracks,'states':states,'model':model,
                    'f0':tracks['f0_hz'].to_numpy(),
                    'duration':states['duration_s'].to_numpy(),
                    'spectral_centroid':tracks['spectral_centroid_hz'].to_numpy()}
