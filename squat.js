// スクワット判定ロジック(MediaPipe Pose の33点ランドマークを入力に使う)
// 推定ライブラリに依存しない純粋な計算なので、Node でも単体テストできる。

export const LM = {
  L_SHOULDER: 11, R_SHOULDER: 12,
  L_HIP: 23, R_HIP: 24,
  L_KNEE: 25, R_KNEE: 26,
  L_ANKLE: 27, R_ANKLE: 28,
};

const SIDES = {
  left: [LM.L_SHOULDER, LM.L_HIP, LM.L_KNEE, LM.L_ANKLE],
  right: [LM.R_SHOULDER, LM.R_HIP, LM.R_KNEE, LM.R_ANKLE],
};

/** b を頂点とした角 a-b-c(度) */
export function angle(a, b, c) {
  const v1x = a.x - b.x, v1y = a.y - b.y;
  const v2x = c.x - b.x, v2y = c.y - b.y;
  const dot = v1x * v2x + v1y * v2y;
  const n = Math.hypot(v1x, v1y) * Math.hypot(v2x, v2y);
  if (n === 0) return NaN;
  return (Math.acos(Math.max(-1, Math.min(1, dot / n))) * 180) / Math.PI;
}

/** 腰→肩のベクトルが鉛直からどれだけ倒れているか(度)。画像座標は y が下向き。 */
export function torsoLean(shoulder, hip) {
  const dx = shoulder.x - hip.x;
  const dy = hip.y - shoulder.y; // 上向きを正に
  return (Math.atan2(Math.abs(dx), dy) * 180) / Math.PI;
}

export const DEFAULTS = {
  standAngle: 160,   // これより膝が伸びたら「立っている」
  descendAngle: 140, // これより曲がったら「しゃがみ始めた」
  depthAngle: 100,   // 膝の最小角度がこれ以下なら十分な深さ
  leanWarn: 55,      // 上体の前傾がこれを超えたら注意
  minRepSec: 1.2,    // 1レップがこれより速ければ「速すぎ」
  minVisibility: 0.5,
  smoothing: 0.4,    // 角度の指数移動平均の係数(大きいほど追従が速い)
};

export const ISSUE_TEXT = {
  shallow: 'もう少し深く',
  lean: '胸を起こして',
  fast: 'ゆっくり下ろして',
};

export class SquatAnalyzer {
  constructor(opts = {}) {
    this.opt = { ...DEFAULTS, ...opts };
    this.reset();
  }

  reset() {
    this.state = 'up';
    this.reps = [];
    this.knee = null;
    this.lean = null;
    this.lastUpTime = null;
    this.cur = null;
    this.lastActiveTime = null;
  }

  /**
   * @param landmarks MediaPipe の landmarks[0](正規化座標 0..1)
   * @param w,h 映像の幅と高さ(縦横比を正しく扱うため)
   * @param t 秒
   * @returns {{visible:boolean, side?:string, knee?:number, hip?:number, lean?:number, state:string, events:object[]}}
   */
  update(landmarks, w, h, t) {
    const events = [];
    if (!landmarks) return { visible: false, state: this.state, events };

    // 体の見えている側を選ぶ(横向き撮影を想定)
    const vis = (ids) => ids.map((i) => landmarks[i]?.visibility ?? 0);
    const score = (ids) => vis(ids).reduce((a, b) => a + b, 0);
    const side = score(SIDES.left) >= score(SIDES.right) ? 'left' : 'right';
    const ids = SIDES[side];
    if (Math.min(...vis(ids)) < this.opt.minVisibility) {
      return { visible: false, state: this.state, events };
    }
    const [sh, hp, kn, an] = ids.map((i) => ({ x: landmarks[i].x * w, y: landmarks[i].y * h }));

    const rawKnee = angle(hp, kn, an);
    const rawHip = angle(sh, hp, kn);
    const rawLean = torsoLean(sh, hp);
    const a = this.opt.smoothing;
    this.knee = this.knee == null ? rawKnee : this.knee + a * (rawKnee - this.knee);
    this.lean = this.lean == null ? rawLean : this.lean + a * (rawLean - this.lean);
    const knee = this.knee, lean = this.lean;

    if (this.state === 'up') {
      if (knee > this.opt.standAngle) this.lastUpTime = t;
      if (knee < this.opt.descendAngle) {
        this.state = 'down';
        this.cur = {
          start: this.lastUpTime ?? t,
          minKnee: knee,
          maxLean: lean,
          leanCued: false,
        };
      }
    } else {
      const c = this.cur;
      c.minKnee = Math.min(c.minKnee, knee);
      c.maxLean = Math.max(c.maxLean, lean);
      if (lean > this.opt.leanWarn && !c.leanCued) {
        c.leanCued = true;
        events.push({ type: 'cue', key: 'lean', text: ISSUE_TEXT.lean });
      }
      if (knee > this.opt.standAngle) {
        const duration = t - c.start;
        const issues = [];
        if (c.minKnee > this.opt.depthAngle) issues.push('shallow');
        if (c.maxLean > this.opt.leanWarn) issues.push('lean');
        if (duration < this.opt.minRepSec) issues.push('fast');
        const rep = {
          n: this.reps.length + 1,
          minKnee: Math.round(c.minKnee),
          maxLean: Math.round(c.maxLean),
          durationSec: Math.round(duration * 10) / 10,
          issues,
        };
        this.reps.push(rep);
        events.push({ type: 'rep', rep });
        this.state = 'up';
        this.lastUpTime = t;
        this.cur = null;
      }
    }
    this.lastActiveTime = t;

    return { visible: true, side, knee, hip: rawHip, lean, state: this.state, events };
  }

  /** Gemini に渡すセットの要約(日本語テキスト) */
  summary() {
    if (!this.reps.length) return 'レップなし';
    const lines = this.reps.map((r) => {
      const iss = r.issues.length ? r.issues.map((k) => ISSUE_TEXT[k]).join('・') : '問題なし';
      return `${r.n}回目: 膝の最小角度${r.minKnee}°、上体の最大前傾${r.maxLean}°、${r.durationSec}秒、${iss}`;
    });
    const good = this.reps.filter((r) => !r.issues.length).length;
    return `スクワット ${this.reps.length}回(問題なし ${good}回)\n` + lines.join('\n');
  }
}
