#!/usr/bin/env python3
"""
main.py — Backend EdgeBench (FastAPI).

  POST /predict   : dự đoán 4 chỉ số bằng CatBoost (đã train sẵn bằng train_models.py)
  WS   /ws/ssh    : SSH tới Jetson Nano, chạy flops_csv2.py, stream output realtime về web

Chạy:  uvicorn main:app --reload --port 8000
Cần:   pip install -r requirements.txt   và đã chạy train_models.py cho cả 2 platform.
"""
import asyncio
import json, hashlib, os, sqlite3, uuid, random, smtplib, ssl, time, tempfile
from collections import OrderedDict
from pathlib import Path as FPath
from dotenv import load_dotenv; load_dotenv()
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import numpy as np, pandas as pd
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from catboost import CatBoostRegressor
from train_models import engineer_features, TARGETS, SEEDS

_FRONTEND = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'index.html')

app = FastAPI(title="EdgeBench API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# --- Email OTP config (đọc từ env khi khởi động uvicorn) ---
MAIL_USER = os.environ.get('MAIL_USER', '')   # vd: edgebench@gmail.com
MAIL_PASS = os.environ.get('MAIL_PASS', '')   # Gmail App Password (16 ký tự)
_pending_otp: dict = {}  # email → {otp, display_name, password_hash, salt, uid, expires_at}

def _send_otp_email(to_email: str, display_name: str, otp: str):
    """Gửi OTP qua Gmail SMTP. Nếu chưa cấu hình env, in ra console (dev mode)."""
    if not MAIL_USER or not MAIL_PASS:
        print(f'[OTP DEV] {to_email} → {otp}')
        return
    msg = MIMEMultipart('alternative')
    msg['Subject'] = f'[EdgeBench] Mã xác nhận đăng ký: {otp}'
    msg['From']    = MAIL_USER
    msg['To']      = to_email
    html = f"""
    <div style="font-family:system-ui,sans-serif;max-width:480px;margin:auto;padding:32px 24px">
      <h2 style="color:#6c63ff;margin-bottom:4px">EdgeBench Predictor</h2>
      <p style="color:#888;font-size:13px;margin-top:0">Hệ thống dự đoán hiệu năng mô hình AI</p>
      <hr style="border:none;border-top:1px solid #eee;margin:20px 0"/>
      <p>Xin chào <strong>{display_name}</strong>,</p>
      <p>Mã xác nhận đăng ký tài khoản của bạn:</p>
      <div style="font-size:40px;font-weight:700;letter-spacing:10px;text-align:center;
                  padding:24px;background:#f8f7ff;border:2px solid #6c63ff;border-radius:12px;
                  color:#6c63ff;margin:20px 0">{otp}</div>
      <p style="color:#555">Mã có hiệu lực trong <strong>10 phút</strong>.</p>
      <p style="color:#aaa;font-size:12px">Nếu bạn không yêu cầu đăng ký, hãy bỏ qua email này.</p>
    </div>"""
    msg.attach(MIMEText(html, 'html', 'utf-8'))
    ctx = ssl.create_default_context()
    try:
        with smtplib.SMTP('smtp.gmail.com', 587, timeout=15) as s:
            s.starttls(context=ctx)
            s.login(MAIL_USER, MAIL_PASS)
            s.sendmail(MAIL_USER, to_email, msg.as_bytes())
        print(f'[MAIL OK] {to_email}')
    except Exception as e:
        print(f'[MAIL ERROR] {to_email} → {e}')
        raise  # để register endpoint biết gửi thất bại

# --- SQLite auth ---
_DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'users.db')

def _db():
    c = sqlite3.connect(_DB); c.row_factory = sqlite3.Row; return c

def _init_db():
    c = _db()
    c.execute("""CREATE TABLE IF NOT EXISTS users (
        id TEXT PRIMARY KEY, email TEXT UNIQUE NOT NULL,
        display_name TEXT DEFAULT '', password_hash TEXT NOT NULL, salt TEXT NOT NULL,
        created_at TEXT DEFAULT (datetime('now')))""")
    c.execute("""CREATE TABLE IF NOT EXISTS sessions (
        token TEXT PRIMARY KEY, user_id TEXT NOT NULL,
        created_at TEXT DEFAULT (datetime('now')))""")
    c.commit(); c.close()

_init_db()

def _hash(pw: str, salt: str) -> str:
    return hashlib.sha256((pw + salt).encode()).hexdigest()

class AuthReq(BaseModel):
    email: str
    password: str
    display_name: str = ''

MAX_PENDING_OTP = 10

@app.post('/auth/register')
async def auth_register(r: AuthReq):
    # Dọn các OTP hết hạn trước khi kiểm tra giới hạn
    now = time.time()
    expired = [e for e, v in _pending_otp.items() if v['expires_at'] < now]
    for e in expired: _pending_otp.pop(e, None)
    if len(_pending_otp) >= MAX_PENDING_OTP:
        raise HTTPException(status_code=429, detail='Hệ thống đang xử lý quá nhiều yêu cầu đăng ký. Vui lòng thử lại sau.')
    # Kiểm tra email đã tồn tại chưa
    c = _db(); row = c.execute('SELECT id FROM users WHERE email=?', (r.email,)).fetchone(); c.close()
    if row:
        raise HTTPException(status_code=400, detail='Email đã được đăng ký.')
    # Tạo OTP và lưu pending (hash password ngay, không lưu plaintext)
    salt = os.urandom(16).hex()
    otp  = f'{random.randint(0, 999999):06d}'
    _pending_otp[r.email] = {
        'otp': otp, 'display_name': r.display_name,
        'password_hash': _hash(r.password, salt), 'salt': salt,
        'uid': str(uuid.uuid4()), 'expires_at': time.time() + 600,
    }
    loop = asyncio.get_running_loop()
    try:
        await loop.run_in_executor(None, _send_otp_email, r.email, r.display_name or r.email, otp)
    except Exception:
        _pending_otp.pop(r.email, None)
        raise HTTPException(status_code=503, detail='Không thể gửi email. Vui lòng thử lại sau.')
    return {'status': 'otp_sent'}

class OtpReq(BaseModel):
    email: str
    otp: str

@app.post('/auth/verify-otp')
def auth_verify_otp(r: OtpReq):
    pending = _pending_otp.get(r.email)
    if not pending:
        raise HTTPException(status_code=400, detail='Không tìm thấy yêu cầu đăng ký. Hãy đăng ký lại.')
    if time.time() > pending['expires_at']:
        _pending_otp.pop(r.email, None)
        raise HTTPException(status_code=400, detail='Mã OTP đã hết hạn. Hãy đăng ký lại.')
    if r.otp != pending['otp']:
        raise HTTPException(status_code=400, detail='Mã OTP không đúng.')
    # OTP hợp lệ → tạo user
    _pending_otp.pop(r.email, None)
    try:
        c = _db()
        c.execute('INSERT INTO users (id,email,display_name,password_hash,salt) VALUES (?,?,?,?,?)',
                  (pending['uid'], r.email, pending['display_name'],
                   pending['password_hash'], pending['salt']))
        c.commit(); c.close()
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail='Email đã được đăng ký.')
    token = str(uuid.uuid4())
    c = _db(); c.execute('INSERT INTO sessions (token,user_id) VALUES (?,?)', (token, pending['uid'])); c.commit(); c.close()
    return {'token': token, 'email': r.email, 'display_name': pending['display_name']}

@app.post('/auth/login')
def auth_login(r: AuthReq):
    c = _db(); row = c.execute('SELECT * FROM users WHERE email=?', (r.email,)).fetchone(); c.close()
    if not row or _hash(r.password, row['salt']) != row['password_hash']:
        raise HTTPException(status_code=401, detail='Email hoặc mật khẩu sai.')
    token = str(uuid.uuid4())
    c = _db(); c.execute('INSERT INTO sessions (token,user_id) VALUES (?,?)', (token, row['id'])); c.commit(); c.close()
    return {'token': token, 'email': row['email'], 'display_name': row['display_name']}

# ----- cache danh sách timm models -----
_TIMM_MODELS: set | None = None
def _get_timm_models() -> set:
    global _TIMM_MODELS
    if _TIMM_MODELS is None:
        try:
            import timm
            _TIMM_MODELS = set(timm.list_models())
        except Exception:
            _TIMM_MODELS = set()
    return _TIMM_MODELS

# ----- nạp model + meta cho mỗi platform -----
STORE = {}
for plat in ['rtx', 'jetson']:
    try:
        meta = json.load(open(f'models/{plat}/meta.json'))
        models = {t: [CatBoostRegressor().load_model(f'models/{plat}/{t}_seed{s}.cbm') for s in SEEDS]
                  for t in TARGETS}
        STORE[plat] = {'meta': meta, 'models': models}
        print(f'[load] {plat}: ok ({len(meta["lookup"])} models)')
    except Exception as e:
        print(f'[load] {plat}: CHƯA có model ({e}) — hãy chạy train_models.py')

class Req(BaseModel):
    platform: str
    model: str
    optimal_bs: float | None = None
    max_bs: float | None = None

class ValidateReq(BaseModel):
    platform: str
    model: str

@app.post("/validate-model")
async def validate_model(r: ValidateReq):
    plat = r.platform if r.platform in STORE else None
    if not plat:
        raise HTTPException(status_code=404, detail=f"Platform '{r.platform}' chưa có model.")
    meta = STORE[plat]['meta']
    name = r.model.strip().lower().split('.')[0]
    in_lookup = name in meta['lookup']
    is_timm = name in _get_timm_models()
    suggestions = []
    if not in_lookup:
        from difflib import get_close_matches
        suggestions = get_close_matches(name, list(meta['lookup'].keys()), n=6, cutoff=0.4)
    return {"name": name, "in_lookup": in_lookup, "is_timm": is_timm, "suggestions": suggestions}

@app.get("/models/{platform}")
async def get_model_list(platform: str):
    if platform not in STORE:
        raise HTTPException(status_code=404, detail=f"Platform '{platform}' chưa có model.")
    keys = sorted(STORE[platform]['meta']['lookup'].keys())
    return {"models": keys, "total": len(keys)}

def _estimate_bs(model_name: str, platform: str = 'rtx') -> int:
    """Ước tính BS mặc định dựa vào size keyword và platform.
    RTX 3080 (10GB VRAM): BS lớn hơn nhiều so với Jetson Nano (4GB shared RAM)."""
    low = model_name.lower()
    is_small = any(k in low for k in ['tiny', 'nano', 'pico', 'atto', 'femto', 'small', 'mini'])
    is_large = any(k in low for k in ['large', 'huge', 'giant', 'xlarge', 'xxlarge', 'xl'])
    if platform == 'jetson':
        if is_small: return 4
        if is_large: return 1
        return 2
    else:  # rtx
        if is_small: return 32
        if is_large: return 8
        return 16

async def _compute_unknown_features(model_name: str) -> dict:
    """Tính đặc trưng tĩnh cho model chưa có trong lookup, dùng timm + torchinfo + ptflops.
    Kết quả được cache để không tính lại cho cùng model."""
    if model_name in _feat_cache:
        return _feat_cache[model_name]

    loop = asyncio.get_running_loop()

    def _compute():
        import timm
        import torchinfo as ti
        import torch
        try:
            from ptflops import get_model_complexity_info
        except ImportError:
            get_model_complexity_info = None

        model = timm.create_model(model_name, pretrained=False).eval()

        # Input size từ default_cfg của timm
        try:
            input_size = int(model.default_cfg['input_size'][-1])
        except Exception:
            input_size = 224

        # torchinfo: params + MACs
        stats = ti.summary(model, input_size=(1, 3, input_size, input_size), verbose=0)
        params_tinfo = int(stats.total_params)
        macs_tinfo = int(stats.total_mult_adds)

        # ptflops (optional)
        try:
            if get_model_complexity_info is None:
                raise ImportError
            macs_pt, params_pt = get_model_complexity_info(
                model, (3, input_size, input_size),
                as_strings=False, print_per_layer_stat=False
            )
        except Exception:
            macs_pt, params_pt = float(macs_tinfo), float(params_tinfo)

        # Activations qua hooks (tổng numel output mỗi layer)
        total_act = [0]
        def _hook(_, __, out):
            if isinstance(out, tuple):
                total_act[0] += sum(o.numel() for o in out if isinstance(o, torch.Tensor))
            else:
                total_act[0] += out.numel()

        hooks = [l.register_forward_hook(_hook) for _, l in model.named_modules()]
        try:
            with torch.no_grad():
                model(torch.randn(1, 3, input_size, input_size))
        except Exception:
            pass
        finally:
            for h in hooks:
                h.remove()

        family = model_name.split('.')[0].split('_')[0].lower()
        return {
            'Model Name': model_name,
            'Input Size': input_size,
            'param_count': params_tinfo,
            'params_tinfo': params_tinfo,
            'macs_tinfo': macs_tinfo,
            'params_ptflops': float(params_pt),
            'macs_ptflops': float(macs_pt),
            'activations': total_act[0],
            'Family': family,
            'Type': 'unknown',
        }

    result = await loop.run_in_executor(None, _compute)
    _feat_cache[model_name] = result
    return result

def _run_catboost(meta, models, row: dict) -> dict:
    df = pd.DataFrame([row])
    feat = engineer_features(df)
    X = feat.reindex(columns=meta['features'])
    for c in meta['numeric']:
        X[c] = pd.to_numeric(X[c], errors='coerce').fillna(meta['medians'].get(c, 0))
    for c in meta['categ']:
        X[c] = X[c].fillna('missing').astype(str)
    out = {}
    for t in TARGETS:
        preds = np.mean([m.predict(X)[0] for m in models[t]])
        if meta['log_targets'][t]:
            preds = np.expm1(preds)
        out[t] = float(np.clip(preds, 0, 100 if t == 'Accuracy' else None))
    return out

@app.post("/predict")
async def predict(r: Req):
    plat = r.platform if r.platform in STORE else None
    if not plat:
        return {"error": f"platform '{r.platform}' chưa có model. Chạy train_models.py."}
    meta, models = STORE[plat]['meta'], STORE[plat]['models']
    bn = r.model.split('.')[0]
    info = meta['lookup'].get(bn)
    from_lookup = info is not None

    if from_lookup:
        row = dict(info)
        if r.optimal_bs: row['Optimal BS'] = r.optimal_bs
        if r.max_bs:     row['Max BS'] = r.max_bs
    else:
        # Model chưa trong dataset → tự tính đặc trưng tĩnh
        opt_bs = r.optimal_bs or _estimate_bs(r.model, plat)
        mx_bs  = r.max_bs or opt_bs
        try:
            static = await _compute_unknown_features(r.model)
        except Exception as e:
            return {"error": f"Không thể tính đặc trưng cho '{r.model}': {e}. "
                             f"Kiểm tra tên model timm hợp lệ (vd: resnet18, vit_tiny_patch16_224)."}
        row = dict(static)
        row['Optimal BS'] = opt_bs
        row['Max BS'] = mx_bs

    loop = asyncio.get_running_loop()
    out = await loop.run_in_executor(None, _run_catboost, meta, models, row)

    return {
        "accuracy": out['Accuracy'], "latency": out['Latency'],
        "throughput": out['Throughput'], "energy": out['Energy'],
        "source": f"CatBoost·{plat}",
        "from_lookup": from_lookup,
    }

@app.post("/predict-file")
async def predict_file(
    platform: str = Form(...),
    file: UploadFile = File(...),
    optimal_bs: float | None = Form(default=None),
    max_bs: float | None = Form(default=None),
):
    ext = FPath(file.filename).suffix.lower() if file.filename else ''
    DIRECT_EXTS    = {'.pth', '.pt'}
    CONVERT_EXTS   = {'.safetensors', '.bin', '.ckpt'}
    if ext not in DIRECT_EXTS | CONVERT_EXTS:
        return {"unsupported_format": True, "ext": ext.lstrip('.')}

    plat = platform if platform in STORE else None
    if not plat:
        return {"error": f"Platform '{platform}' chưa có model."}

    content = await file.read()
    if len(content) > 2 * 1024 * 1024 * 1024:
        return {"error": "File quá lớn (giới hạn 2 GB)."}

    tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    tmp.write(content); tmp.close()
    tmp_path = tmp.name

    try:
        meta, mdls = STORE[plat]['meta'], STORE[plat]['models']
        stem = FPath(file.filename).stem.lower()
        opt_bs_val = optimal_bs or _estimate_bs(stem, plat)
        mx_bs_val  = max_bs or opt_bs_val
        loop = asyncio.get_running_loop()

        def _load_and_extract():
            try:
                import torch
            except ImportError:
                raise ValueError("torch chưa cài — không thể đọc .pth.")

            # Ưu tiên khớp tên file với lookup trước khi đọc nội dung
            lookup = meta['lookup']
            if stem in lookup:
                info = lookup[stem]
                features = dict(info)
                features['Optimal BS'] = opt_bs_val
                features['Max BS']     = mx_bs_val
                return features, f'name_match:{stem}', info.get('param_count') or info.get('params_tinfo', 0)

            # ── Load file theo định dạng ────────────────────────────────────
            if ext == '.safetensors':
                try:
                    from safetensors.torch import load_file as sf_load
                except ImportError:
                    raise ValueError("safetensors chưa cài — pip install safetensors")
                data = sf_load(tmp_path)   # trả về dict[str, Tensor]

            elif ext == '.ckpt':
                # PyTorch Lightning checkpoint — có thể chứa key 'state_dict'
                try:
                    data = torch.load(tmp_path, map_location='cpu', weights_only=True)
                except Exception:
                    data = torch.load(tmp_path, map_location='cpu', weights_only=False)
                for key in ('state_dict', 'model_state_dict', 'model', 'net'):
                    if isinstance(data, dict) and key in data:
                        data = data[key]; break

            else:
                # .pth / .pt / .bin — torch.load xử lý được hết
                try:
                    data = torch.load(tmp_path, map_location='cpu', weights_only=True)
                except Exception:
                    data = torch.load(tmp_path, map_location='cpu', weights_only=False)

            # ── Case 1: full nn.Module — dùng torchinfo tính MACs chính xác ─
            if isinstance(data, torch.nn.Module):
                import torchinfo as ti
                model = data.eval()
                try:
                    input_size = int(model.default_cfg['input_size'][-1])
                except Exception:
                    input_size = 224

                stats = ti.summary(model, input_size=(1, 3, input_size, input_size), verbose=0)
                total_params = int(stats.total_params)
                macs = int(stats.total_mult_adds)

                total_act = [0]
                def _hook(_, __, out):
                    if isinstance(out, tuple):
                        total_act[0] += sum(o.numel() for o in out if isinstance(o, torch.Tensor))
                    else:
                        total_act[0] += out.numel()
                hooks = [l.register_forward_hook(_hook) for _, l in model.named_modules()]
                try:
                    with torch.no_grad():
                        model(torch.randn(1, 3, input_size, input_size))
                except Exception:
                    pass
                finally:
                    for h in hooks: h.remove()

                # Thử khớp lookup trước (fine-tuned model có params giống base)
                lookup = meta['lookup']
                best_name, best_info, best_diff = None, None, float('inf')
                for mname, info in lookup.items():
                    lp = info.get('param_count') or info.get('params_tinfo', 0)
                    if lp > 0:
                        diff = abs(lp - total_params) / max(lp, total_params)
                        if diff < best_diff:
                            best_diff, best_name, best_info = diff, mname, info
                if best_diff < 0.02:
                    features = dict(best_info)
                    features['Optimal BS'] = opt_bs_val
                    features['Max BS']     = mx_bs_val
                    return features, f'param_match:{best_name}:{best_diff*100:.2f}%', total_params

                # Không khớp lookup → dùng features tính chính xác từ torchinfo
                family = stem.split('_')[0]
                features = {
                    'Model Name': stem, 'Input Size': input_size,
                    'param_count': total_params, 'params_tinfo': total_params,
                    'macs_tinfo': macs, 'params_ptflops': float(total_params),
                    'macs_ptflops': float(macs), 'activations': total_act[0],
                    'Family': family, 'Type': 'unknown',
                    'Optimal BS': opt_bs_val, 'Max BS': mx_bs_val,
                }
                return features, 'full_model', total_params

            # ── Case 2: state dict / checkpoint ────────────────────────────
            sd = data
            if isinstance(data, dict):
                for key in ('model', 'state_dict', 'model_state_dict', 'net', 'network'):
                    if key in data and isinstance(data[key], (dict, OrderedDict)):
                        sd = data[key]; break

            if not isinstance(sd, (dict, OrderedDict)):
                raise ValueError("Không nhận dạng được định dạng file — không phải model hay state dict.")

            BUFFERS = ('running_mean', 'running_var', 'num_batches_tracked',
                       'total_ops', 'total_params')
            param_tensors = {
                k: v for k, v in sd.items()
                if isinstance(v, torch.Tensor)
                and not any(k.endswith(b) for b in BUFFERS)
            }
            total_params = sum(t.numel() for t in param_tensors.values())
            if total_params == 0:
                raise ValueError("Không đọc được tham số — file có thể rỗng hoặc sai định dạng.")

            # Try to match by param count in lookup (within 2%)
            lookup = meta['lookup']
            best_name, best_info, best_diff = None, None, float('inf')
            for mname, info in lookup.items():
                lp = info.get('param_count') or info.get('params_tinfo', 0)
                if lp > 0:
                    diff = abs(lp - total_params) / max(lp, total_params)
                    if diff < best_diff:
                        best_diff, best_name, best_info = diff, mname, info

            if best_diff < 0.02:
                features = dict(best_info)
                features['Optimal BS'] = opt_bs_val
                features['Max BS']     = mx_bs_val
                return features, f'param_match:{best_name}:{best_diff*100:.2f}%', total_params

            # Rough estimation — no confident match
            has_attn = any('attn' in k or 'attention' in k for k in param_tensors)
            has_conv = any('conv' in k for k in param_tensors)
            macs_est = int(total_params * (1.5 if (has_attn and not has_conv) else 2.0))
            features = {
                'Model Name': stem, 'Input Size': 224,
                'param_count': total_params, 'params_tinfo': total_params,
                'macs_tinfo': macs_est, 'params_ptflops': float(total_params),
                'macs_ptflops': float(macs_est), 'activations': total_params // 8,
                'Family': stem.split('_')[0], 'Type': 'unknown',
                'Optimal BS': opt_bs_val, 'Max BS': mx_bs_val,
            }
            return features, f'estimated', total_params

        features, note, params = await loop.run_in_executor(None, _load_and_extract)
        out = await loop.run_in_executor(None, _run_catboost, meta, mdls, features)

        matched = note.split(':')[1] if note.startswith('param_match:') else None
        return {
            "accuracy": out['Accuracy'], "latency": out['Latency'],
            "throughput": out['Throughput'], "energy": out['Energy'],
            "match_note": note, "param_count": params, "matched_model": matched,
        }

    except Exception as e:
        return {"error": str(e)}
    finally:
        try: os.unlink(tmp_path)
        except Exception: pass

# ----------------- Shared SSH session store -----------------
_sessions: dict = {}
_feat_cache: dict = {}  # model_name → static features (tránh tính lại mỗi request)
# session: {buffer, subs, done, channel, client, task}

async def _bcast(sess: dict, line: str):
    sess['buffer'].append(line)
    dead = set()
    for sub in list(sess['subs']):
        try:
            await sub.send_text(line)
        except Exception:
            dead.add(sub)
    sess['subs'] -= dead

async def _kill_session(token: str):
    if not token or token not in _sessions:
        return
    sess = _sessions[token]
    if sess.get('task') and not sess['task'].done():
        sess['task'].cancel()
    try:
        sess['channel'].close()
        sess['client'].exec_command("pkill -f flops_csv2.py")
    except Exception:
        pass
    sess['done'] = True
    await _bcast(sess, "[dừng] Benchmark đã bị dừng.")
    _sessions.pop(token, None)

# ----------------- WebSocket SSH terminal -----------------
@app.websocket("/ws/ssh")
async def ws_ssh(ws: WebSocket):
    await ws.accept()
    token = None
    try:
        cfg = json.loads(await ws.receive_text())
        token = cfg.get('token', '')

        # --- Reconnect to existing session ---
        if cfg.get('reconnect'):
            if token and token in _sessions and not _sessions[token]['done']:
                sess = _sessions[token]
                sess['subs'].add(ws)
                for line in list(sess['buffer']):
                    try:
                        await ws.send_text(line)
                    except Exception:
                        break
                while not sess['done']:
                    try:
                        msg = await asyncio.wait_for(ws.receive_text(), timeout=1.0)
                        if msg == 'pause':
                            try:
                                sess['client'].exec_command("kill -STOP $(pgrep -f flops_csv2.py)")
                            except Exception:
                                pass
                            sess['paused'] = True
                            await _bcast(sess, '__PAUSED__')
                        elif msg == 'resume':
                            try:
                                sess['client'].exec_command("kill -CONT $(pgrep -f flops_csv2.py)")
                            except Exception:
                                pass
                            sess['paused'] = False
                            await _bcast(sess, '__RESUMED__')
                        elif msg in ('stop', 'kill'):
                            await _kill_session(token)
                    except asyncio.TimeoutError:
                        pass
                    except WebSocketDisconnect:
                        break
                sess['subs'].discard(ws)
            else:
                try:
                    await ws.send_text('__no_session__')
                except Exception:
                    pass
            return

        # --- Start new SSH session ---
        import paramiko
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(cfg['host'], username=cfg['user'],
                       password=cfg.get('password'), timeout=15)

        model = cfg.get('model', 'resnet18')
        cmd = f"cd ~/edgebench && python3 flops_csv2.py --model {model} 2>&1"
        stdin, stdout_ch, _ = client.exec_command(cmd, get_pty=True)

        sess = {
            'buffer': [], 'subs': {ws}, 'done': False, 'paused': False,
            'channel': stdout_ch.channel, 'client': client, 'task': None,
        }
        if token:
            _sessions[token] = sess

        await _bcast(sess, f"$ {cmd}")

        loop = asyncio.get_running_loop()

        async def read_loop():
            try:
                while True:
                    line = await loop.run_in_executor(None, stdout_ch.readline)
                    if not line:
                        break
                    await _bcast(sess, line.rstrip('\n'))
            except asyncio.CancelledError:
                pass
            except Exception as ex:
                await _bcast(sess, f"[lỗi đọc] {ex}")
            finally:
                if not sess['done']:
                    await _bcast(sess, "✓ Hoàn tất — đã ghi vào JetsonNano_model.csv")
                sess['done'] = True
                _sessions.pop(token, None)
                try:
                    client.close()
                except Exception:
                    pass

        task = asyncio.create_task(read_loop())
        sess['task'] = task

        try:
            while not sess['done']:
                try:
                    msg = await asyncio.wait_for(ws.receive_text(), timeout=0.5)
                    if msg == 'pause':
                        try:
                            sess['client'].exec_command("kill -STOP $(pgrep -f flops_csv2.py)")
                        except Exception:
                            pass
                        sess['paused'] = True
                        await _bcast(sess, '__PAUSED__')
                    elif msg == 'resume':
                        try:
                            sess['client'].exec_command("kill -CONT $(pgrep -f flops_csv2.py)")
                        except Exception:
                            pass
                        sess['paused'] = False
                        await _bcast(sess, '__RESUMED__')
                    elif msg in ('stop', 'kill'):
                        await _kill_session(token)
                except asyncio.TimeoutError:
                    continue
                except WebSocketDisconnect:
                    break
        finally:
            sess['subs'].discard(ws)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await ws.send_text(f"[lỗi] {e}")
        except Exception:
            pass
    finally:
        try:
            await ws.close()
        except Exception:
            pass

@app.get("/ssh/status")
def ssh_status(token: str = ""):
    if token and token in _sessions and not _sessions[token]['done']:
        sess = _sessions[token]
        return {"active": True, "paused": sess.get('paused', False), "lines": len(sess['buffer'])}
    return {"active": False, "paused": False}

@app.get("/health")
def health():
    def _check(pkg):
        try: __import__(pkg); return True
        except ImportError: return False
    return {
        "ok": True,
        "platforms_loaded": list(STORE.keys()),
        "packages": {
            "torch": _check("torch"),
            "timm": _check("timm"),
            "torchinfo": _check("torchinfo"),
            "ptflops": _check("ptflops"),
            "python_multipart": _check("multipart"),
        }
    }

@app.get("/")
def root(): return FileResponse(_FRONTEND)
