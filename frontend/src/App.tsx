import { useEffect, useMemo, useRef, useState, type CSSProperties, type DragEvent } from "react";

type SourceTab = "youtube" | "local";
type TemplateId = "viral" | "podcast" | "clean" | "custom";
type PortraitMode = "full" | "letterbox" | "blur";

type Clip = {
  id: string;
  title: string;
  reason: string;
  hook: string;
  start: number;
  end: number;
  score: number;
  selected: boolean;
};

type Job = {
  id: string;
  status: string;
  step: string;
  error: string | None;
  title: string;
  duration: number;
  clips: Clip[];
  exports: { name: string; url: string }[];
  video_url: string | null;
  lines?: { start: number; end: number; text: string }[];
};

type ClipStyle = {
  template: TemplateId;
  font: string;
  position: "center" | "bottom";
  active_color: string;
  idle_color: string;
  caption_size: "small" | "normal" | "large";
  crop: "tight" | "normal" | "loose";
  quality: "fast" | "good";
  watermark: string;
  zoom_punch: boolean;
  hook: boolean;
  fade: boolean;
  safe_margin: number;
  blur_amount: number;
};

type None = null;

const PRESETS = [15, 30, 45, 60];
const COUNTS = [1, 3, 5, 8];
const FONTS = ["Arial Black", "Arial", "Impact"];

function lockAt(rel: number, pts: { t: number; x: number }[]): number {
  if (!pts.length) return 0.5;
  let chosen = pts[0].x;
  for (const pt of pts) {
    if (pt.t <= rel) chosen = pt.x;
    else break;
  }
  return chosen;
}

const TEMPLATES: Record<Exclude<TemplateId, "custom">, ClipStyle> = {
  viral: {
    template: "viral",
    font: "Arial Black",
    position: "center",
    active_color: "#FFFF00",
    idle_color: "#F0F0F0",
    caption_size: "large",
    crop: "tight",
    quality: "good",
    watermark: "",
    zoom_punch: true,
    hook: false,
    fade: true,
    safe_margin: 140,
    blur_amount: 55,
  },
  podcast: {
    template: "podcast",
    font: "Arial",
    position: "bottom",
    active_color: "#7CFFB2",
    idle_color: "#E8E8E8",
    caption_size: "normal",
    crop: "loose",
    quality: "good",
    watermark: "",
    zoom_punch: false,
    hook: false,
    fade: true,
    safe_margin: 90,
    blur_amount: 40,
  },
  clean: {
    template: "clean",
    font: "Arial",
    position: "bottom",
    active_color: "#FFFFFF",
    idle_color: "#D0D0D0",
    caption_size: "small",
    crop: "normal",
    quality: "fast",
    watermark: "",
    zoom_punch: false,
    hook: false,
    fade: true,
    safe_margin: 80,
    blur_amount: 30,
  },
};

async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, init);
  if (!res.ok) {
    let message = res.statusText;
    try {
      const body = await res.json();
      message = body.detail || body.error || message;
    } catch {
      message = await res.text();
    }
    throw new Error(typeof message === "string" ? message : JSON.stringify(message));
  }
  return res.json() as Promise<T>;
}

function formatTime(total: number) {
  const value = Math.max(0, total);
  const m = Math.floor(value / 60);
  const s = Math.floor(value % 60);
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

function clipDuration(clip: Clip) {
  return Math.max(0, clip.end - clip.start);
}

function stamp() {
  return new Date().toLocaleTimeString("id-ID", { hour12: false });
}

function PreviewOverlays({
  style,
  caption,
  active,
  playTime,
  lines,
}: {
  style: ClipStyle;
  caption: boolean;
  active: Clip | null;
  playTime: number;
  lines: { start: number; end: number; text: string }[];
}) {
  const line =
    lines.find((item) => playTime >= item.start && playTime < item.end)?.text ||
    (caption ? active?.title || "" : "");
  const hook = style.hook ? active?.hook || active?.title || "" : "";
  if (!hook && !line && !style.watermark) return null;
  return (
    <div className={`ov ${style.position}`} aria-hidden="true">
      {hook ? <div className="ov-hook">{hook}</div> : null}
      {caption && line ? (
        <div className="ov-lyric" style={{ color: style.active_color }}>
          {line}
        </div>
      ) : null}
      {style.watermark ? <div className="ov-mark">{style.watermark}</div> : null}
    </div>
  );
}

export default function App() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const blurRef = useRef<HTMLVideoElement>(null);
  const logRef = useRef<HTMLDivElement>(null);
  const [tab, setTab] = useState<SourceTab>("youtube");
  const [url, setUrl] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [preset, setPreset] = useState<number | "custom">(30);
  const [customDuration, setCustomDuration] = useState(20);
  const [clipCount, setClipCount] = useState(3);
  const [orientation, setOrientation] = useState<"portrait" | "landscape">("portrait");
  const [portraitMode, setPortraitMode] = useState<PortraitMode>("full");
  const [caption, setCaption] = useState(true);
  const [focusSpeaker, setFocusSpeaker] = useState(true);
  const [style, setStyle] = useState<ClipStyle>(TEMPLATES.viral);
  const [showCustomize, setShowCustomize] = useState(false);
  const [job, setJob] = useState<Job | null>(null);
  const [clips, setClips] = useState<Clip[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [hint, setHint] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [logs, setLogs] = useState<string[]>(["[boot] CLIPER // CLIP STUDIO online"]);
  const [lockX, setLockX] = useState<number | null>(null);
  const [lockPts, setLockPts] = useState<{ t: number; x: number }[]>([]);
  const [playTime, setPlayTime] = useState(0);

  const targetDuration = preset === "custom" ? customDuration : preset;
  const working = job?.status === "working" || job?.status === "exporting" || busy;
  const active = clips.find((c) => c.id === activeId) || null;
  const selected = useMemo(() => clips.filter((c) => c.selected), [clips]);
  const phase = !job ? 1 : clips.length === 0 ? 2 : job.exports.length ? 4 : 3;
  const badge =
    job?.status === "error" || error
      ? "ERROR"
      : working
        ? "RUNNING"
        : job?.status === "ready"
          ? "READY"
          : "IDLE";

  function pushLog(line: string) {
    setLogs((prev) => {
      if (prev[prev.length - 1]?.includes(line)) return prev;
      return [...prev.slice(-48), `[${stamp()}] ${line}`];
    });
  }

  function applyTemplate(id: TemplateId) {
    if (id === "custom") {
      setStyle((prev) => ({ ...prev, template: "custom" }));
      setShowCustomize(true);
      return;
    }
    setStyle(TEMPLATES[id]);
    setShowCustomize(false);
  }

  function patchStyle<K extends keyof ClipStyle>(key: K, value: ClipStyle[K]) {
    setStyle((prev) => ({ ...prev, [key]: value, template: "custom" }));
  }

  useEffect(() => {
    if (job?.step) pushLog(job.step);
  }, [job?.step]);

  useEffect(() => {
    logRef.current?.scrollTo({ top: logRef.current.scrollHeight });
  }, [logs, working]);

  useEffect(() => {
    if (!job || (job.status !== "working" && job.status !== "exporting")) return;
    const timer = window.setInterval(async () => {
      try {
        const next = await api<Job>(`/api/jobs/${job.id}`);
        setJob(next);
        if (next.clips?.length) setClips(next.clips);
        if (next.status === "error") setError(next.error);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Gagal memuat status");
      }
    }, 1500);
    return () => window.clearInterval(timer);
  }, [job?.id, job?.status]);

  useEffect(() => {
    if (!job || orientation !== "portrait" || portraitMode !== "full") {
      setLockX(null);
      setLockPts([]);
      return;
    }
    let cancelled = false;
    const start = active?.start ?? 0;
    const end = active?.end ?? Math.min(Number(job.duration) || 8, start + 8);
    void api<{ x: number; locked: boolean; points?: { t: number; x: number }[] }>("/api/frame", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ job_id: job.id, start, end }),
    })
      .then((frame) => {
        if (cancelled) return;
        if (!frame.locked) {
          setLockX(null);
          setLockPts([]);
          return;
        }
        setLockPts(frame.points || []);
        setLockX(frame.x);
      })
      .catch(() => {
        if (!cancelled) {
          setLockX(null);
          setLockPts([]);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [job?.id, active?.id, orientation, portraitMode]);

  useEffect(() => {
    const video = videoRef.current;
    if (!video || !active) return;
    const onTime = () => {
      if (video.currentTime >= active.end - 0.05) {
        video.pause();
        video.currentTime = active.end;
      }
    };
    video.addEventListener("timeupdate", onTime);
    return () => video.removeEventListener("timeupdate", onTime);
  }, [active]);

  async function ingest() {
    setError(null);
    const line = tab === "youtube" ? "Mengunduh video dari YouTube..." : "Mengunggah video lokal...";
    setHint(line);
    pushLog(line);
    setBusy(true);
    try {
      let next: Job;
      if (tab === "youtube") {
        if (!url.trim()) throw new Error("Tempel URL YouTube dulu");
        next = await api<Job>("/api/ingest/youtube", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ url: url.trim() }),
        });
      } else {
        if (!file) throw new Error("Pilih file video dulu");
        const data = new FormData();
        data.append("file", file);
        next = await api<Job>("/api/ingest/upload", { method: "POST", body: data });
      }
      setJob(next);
      setClips([]);
      setActiveId(null);
      pushLog(`Sumber siap: ${next.title || next.id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Ingest gagal");
    } finally {
      setBusy(false);
    }
  }

  async function analyze() {
    if (!job) return;
    setError(null);
    setHint("AI mencari momen terbaik...");
    pushLog(`Analyze ${clipCount} clip @ ${targetDuration}s`);
    setBusy(true);
    try {
      const next = await api<Job>("/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          job_id: job.id,
          target_duration: targetDuration,
          clip_count: clipCount,
        }),
      });
      setJob(next);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Analisis gagal");
    } finally {
      setBusy(false);
    }
  }

  async function generateVideos() {
    if (!job) return;
    setError(null);
    setHint("Menyiapkan generate video...");
    pushLog(`Export template=${style.template} crop=${style.crop}`);
    setBusy(true);
    try {
      const next = await api<Job>("/api/export", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          job_id: job.id,
          fade: style.fade,
          caption,
          hook: style.hook,
          orientation,
          portrait_mode: orientation === "portrait" ? portraitMode : "full",
          focus_speaker: focusSpeaker && portraitMode === "full",
          style,
          clips: selected.length ? selected : clips,
        }),
      });
      setJob(next);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Generate gagal");
    } finally {
      setBusy(false);
    }
  }

  function playClip(clip: Clip) {
    setActiveId(clip.id);
    const video = videoRef.current;
    if (!video) return;
    video.currentTime = clip.start;
    void video.play();
  }

  const loadingTitle =
    job?.status === "exporting"
      ? "GENERATE // RENDER"
      : job?.status === "working"
        ? "ANALYZE // SELECT"
        : busy
          ? "PROCESS"
          : "";
  const loadingDetail = job?.step || hint || "Mohon tunggu sebentar";
  const loadingStages = [
    { key: "unduh", label: "INGEST", on: Boolean(job?.video_url) || hint.includes("Mengunduh") || hint.includes("Mengunggah") },
    { key: "transkrip", label: "TRANSCRIBE + AI", on: job?.status === "working" || clips.length > 0 },
    { key: "export", label: "SPEAKER LOCK + CAPTION", on: job?.status === "exporting" || Boolean(job?.exports.length) },
    { key: "selesai", label: "READY", on: Boolean(job?.exports.length) && job?.status === "ready" },
  ];

  function onDrop(event: DragEvent) {
    event.preventDefault();
    setDragOver(false);
    const next = event.dataTransfer.files[0];
    if (next) {
      setFile(next);
      setTab("local");
    }
  }

  return (
    <div className="page">
      {working && (
        <div className="loader" role="status" aria-live="polite">
          <div className="loader-card">
            <div className="scanlines" aria-hidden="true" />
            <p className="loader-kicker">SYS // PROCESS</p>
            <h3>{loadingTitle}</h3>
            <p className="loader-detail">{loadingDetail}</p>
            {job?.title ? <p className="loader-meta">{job.title}</p> : null}
            <div className="bar" aria-hidden="true">
              <span />
            </div>
            <ul className="loader-stages">
              {loadingStages.map((stage) => (
                <li key={stage.key} className={stage.on ? "on" : ""}>
                  <i />
                  {stage.label}
                </li>
              ))}
            </ul>
            <div className="proc-log" ref={logRef}>
              {logs.slice(-8).map((line) => (
                <div key={line}>{line}</div>
              ))}
            </div>
          </div>
        </div>
      )}

      <header className="topbar">
        <div>
          <p className="eyebrow">LOCAL CLIP PIPELINE</p>
          <h1>
            CLIPER <span>// CLIP STUDIO</span>
          </h1>
        </div>
        <div className="top-meta">
          <span className={`badge-sys ${badge.toLowerCase()}`}>{badge}</span>
          <ol className="steps">
            {["SRC", "AI", "GEN", "DL"].map((label, index) => (
              <li key={label} className={phase >= index + 1 ? "on" : ""}>
                <span>0{index + 1}</span>
                {label}
              </li>
            ))}
          </ol>
        </div>
      </header>

      <section className="grid-main">
        <div className="stack">
          <section className="panel">
            <div className="panel-head">
              <h2>01 // SUMBER</h2>
              <div className="tabs">
                <button className={tab === "youtube" ? "on" : ""} onClick={() => setTab("youtube")}>
                  YOUTUBE
                </button>
                <button className={tab === "local" ? "on" : ""} onClick={() => setTab("local")}>
                  LOKAL
                </button>
              </div>
            </div>

            {tab === "youtube" ? (
              <div className="row">
                <input
                  value={url}
                  onChange={(e) => setUrl(e.target.value)}
                  placeholder="https://www.youtube.com/watch?v=..."
                />
                <button className="primary" disabled={working} onClick={() => void ingest()}>
                  AMBIL
                </button>
              </div>
            ) : (
              <>
                <label
                  className={`drop ${dragOver ? "over" : ""}`}
                  onDragOver={(e) => {
                    e.preventDefault();
                    setDragOver(true);
                  }}
                  onDragLeave={() => setDragOver(false)}
                  onDrop={onDrop}
                >
                  <input
                    type="file"
                    accept="video/mp4,video/quicktime,video/webm,video/x-matroska,.mp4,.mov,.mkv,.webm"
                    hidden
                    onChange={(e) => setFile(e.target.files?.[0] || null)}
                  />
                  <strong>{file ? file.name : "DROP VIDEO / KLIK PILIH"}</strong>
                  <span>mp4 · mov · mkv · webm</span>
                </label>
                <button className="primary" disabled={working || !file} onClick={() => void ingest()}>
                  UPLOAD
                </button>
              </>
            )}
          </section>

          <section className="panel">
            <div className="panel-head">
              <h2>02 // TEMPLATE + AI</h2>
            </div>
            <div className="setting">
              <span>LOOK</span>
              <div className="presets">
                {(["viral", "podcast", "clean", "custom"] as TemplateId[]).map((id) => (
                  <button key={id} className={style.template === id ? "on" : ""} onClick={() => applyTemplate(id)}>
                    {id.toUpperCase()}
                  </button>
                ))}
              </div>
            </div>
            <div className="setting">
              <span>OUTPUT DARI 1 VIDEO</span>
              <div className="presets">
                {COUNTS.map((value) => (
                  <button key={value} className={clipCount === value ? "on" : ""} onClick={() => setClipCount(value)}>
                    {value} VID
                  </button>
                ))}
              </div>
            </div>
            <div className="setting">
              <span>DURASI CLIP</span>
              <div className="presets">
                {PRESETS.map((value) => (
                  <button key={value} className={preset === value ? "on" : ""} onClick={() => setPreset(value)}>
                    {value}s
                  </button>
                ))}
                <button className={preset === "custom" ? "on" : ""} onClick={() => setPreset("custom")}>
                  CUSTOM
                </button>
                {preset === "custom" && (
                  <input
                    className="tiny"
                    type="number"
                    min={8}
                    max={180}
                    value={customDuration}
                    onChange={(e) => setCustomDuration(Number(e.target.value) || 15)}
                  />
                )}
              </div>
            </div>
            <div className="setting">
              <span>ORIENTASI</span>
              <div className="presets">
                <button className={orientation === "portrait" ? "on" : ""} onClick={() => setOrientation("portrait")}>
                  9:16
                </button>
                <button className={orientation === "landscape" ? "on" : ""} onClick={() => setOrientation("landscape")}>
                  16:9
                </button>
              </div>
            </div>
            {orientation === "portrait" && (
              <div className="setting">
                <span>MODE PORTRAIT</span>
                <div className="presets">
                  <button className={portraitMode === "full" ? "on" : ""} onClick={() => setPortraitMode("full")}>
                    FULL
                  </button>
                  <button className={portraitMode === "letterbox" ? "on" : ""} onClick={() => setPortraitMode("letterbox")}>
                    FIT
                  </button>
                  <button className={portraitMode === "blur" ? "on" : ""} onClick={() => setPortraitMode("blur")}>
                    BLUR BG
                  </button>
                </div>
                {portraitMode === "blur" && (
                  <label className="blur-slider">
                    <span>BLUR {style.blur_amount}</span>
                    <input
                      type="range"
                      min={0}
                      max={100}
                      value={style.blur_amount}
                      onChange={(e) => patchStyle("blur_amount", Number(e.target.value))}
                    />
                    <em>halus</em>
                    <em>kuat</em>
                  </label>
                )}
              </div>
            )}
            <div className="toggles">
              <label>
                <input type="checkbox" checked={caption} onChange={(e) => setCaption(e.target.checked)} />
                CAPTION
              </label>
              <label>
                <input type="checkbox" checked={style.fade} onChange={(e) => patchStyle("fade", e.target.checked)} />
                FADE
              </label>
              <label>
                <input type="checkbox" checked={style.hook} onChange={(e) => patchStyle("hook", e.target.checked)} />
                HOOK
              </label>
              <label>
                <input type="checkbox" checked={focusSpeaker} onChange={(e) => setFocusSpeaker(e.target.checked)} />
                LOCK SPEAKER
              </label>
              <label>
                <input
                  type="checkbox"
                  checked={style.zoom_punch}
                  onChange={(e) => patchStyle("zoom_punch", e.target.checked)}
                />
                ZOOM PUNCH
              </label>
            </div>
            <button className="ghost" type="button" onClick={() => setShowCustomize((v) => !v)}>
              {showCustomize ? "TUTUP CUSTOMIZE" : "BUKA CUSTOMIZE"}
            </button>
            {showCustomize && (
              <div className="customize">
                <div className="setting">
                  <span>FONT</span>
                  <div className="presets">
                    {FONTS.map((font) => (
                      <button key={font} className={style.font === font ? "on" : ""} onClick={() => patchStyle("font", font)}>
                        {font}
                      </button>
                    ))}
                  </div>
                </div>
                <div className="setting">
                  <span>POSISI CAPTION</span>
                  <div className="presets">
                    <button className={style.position === "center" ? "on" : ""} onClick={() => patchStyle("position", "center")}>
                      CENTER
                    </button>
                    <button className={style.position === "bottom" ? "on" : ""} onClick={() => patchStyle("position", "bottom")}>
                      BOTTOM
                    </button>
                  </div>
                </div>
                <div className="setting">
                  <span>UKURAN</span>
                  <div className="presets">
                    {(["small", "normal", "large"] as const).map((size) => (
                      <button
                        key={size}
                        className={style.caption_size === size ? "on" : ""}
                        onClick={() => patchStyle("caption_size", size)}
                      >
                        {size.toUpperCase()}
                      </button>
                    ))}
                  </div>
                </div>
                <div className="setting">
                  <span>CROP</span>
                  <div className="presets">
                    {(["tight", "normal", "loose"] as const).map((crop) => (
                      <button key={crop} className={style.crop === crop ? "on" : ""} onClick={() => patchStyle("crop", crop)}>
                        {crop.toUpperCase()}
                      </button>
                    ))}
                  </div>
                </div>
                <div className="setting">
                  <span>QUALITY</span>
                  <div className="presets">
                    <button className={style.quality === "fast" ? "on" : ""} onClick={() => patchStyle("quality", "fast")}>
                      FAST
                    </button>
                    <button className={style.quality === "good" ? "on" : ""} onClick={() => patchStyle("quality", "good")}>
                      GOOD
                    </button>
                  </div>
                </div>
                <div className="color-row">
                  <label>
                    ACTIVE
                    <input type="color" value={style.active_color} onChange={(e) => patchStyle("active_color", e.target.value)} />
                  </label>
                  <label>
                    IDLE
                    <input type="color" value={style.idle_color} onChange={(e) => patchStyle("idle_color", e.target.value)} />
                  </label>
                </div>
                <input
                  value={style.watermark}
                  onChange={(e) => patchStyle("watermark", e.target.value)}
                  placeholder="Watermark teks (opsional)"
                />
              </div>
            )}
            <button className="primary wide" disabled={working || !job} onClick={() => void analyze()}>
              AI PILIH {clipCount} CLIP
            </button>
          </section>
        </div>

        <aside className="stack">
          <div className="preview-row">
            <div
              className={`player-wrap ${orientation} ${orientation === "portrait" ? portraitMode : ""} ${
                lockX != null ? "locked" : ""
              }`}
              style={
                {
                  "--bg-blur": `${4 + style.blur_amount * 0.28}px`,
                  "--lock-x": lockX != null ? `${Math.round(lockX * 100)}%` : "50%",
                } as CSSProperties
              }
            >
              {job?.video_url && orientation === "portrait" && portraitMode === "blur" ? (
                <video
                  ref={blurRef}
                  className="bg-blur"
                  src={job.video_url}
                  muted
                  playsInline
                  aria-hidden="true"
                />
              ) : null}
              {job?.video_url ? (
                <video
                  ref={videoRef}
                  className="fg"
                  src={job.video_url}
                  controls
                  playsInline
                  onPlay={() => {
                    const bg = blurRef.current;
                    if (bg) {
                      bg.currentTime = videoRef.current?.currentTime || 0;
                      void bg.play();
                    }
                  }}
                  onPause={() => blurRef.current?.pause()}
                  onSeeked={(e) => {
                    const bg = blurRef.current;
                    if (bg) bg.currentTime = e.currentTarget.currentTime;
                  }}
                  onTimeUpdate={(e) => {
                    const t = e.currentTarget.currentTime;
                    setPlayTime(t);
                    if (lockPts.length) {
                      setLockX(lockAt(t - (active?.start ?? 0), lockPts));
                    }
                    const bg = blurRef.current;
                    if (bg && Math.abs(bg.currentTime - t) > 0.25) {
                      bg.currentTime = t;
                    }
                  }}
                />
              ) : (
                <div className="placeholder">NO SIGNAL // TUNGGU SUMBER</div>
              )}
              <PreviewOverlays
                style={style}
                caption={caption}
                active={active}
                playTime={playTime}
                lines={job?.lines || []}
              />
            </div>
            {orientation === "portrait" && (
              <div className="portrait-modes">
                {(
                  [
                    ["full", "FULL", "Landscape jadi portrait"],
                    ["letterbox", "FIT", "Versi sekarang"],
                    ["blur", "BLUR", "BG video blur"],
                  ] as const
                ).map(([id, label, hint]) => (
                  <button
                    key={id}
                    type="button"
                    className={portraitMode === id ? "on" : ""}
                    onClick={() => setPortraitMode(id)}
                  >
                    <b>{label}</b>
                    <span>{hint}</span>
                  </button>
                ))}
              </div>
            )}
          </div>
          {job && (
            <p className="status">
              <b>{job.step}</b>
              {job.title ? ` · ${job.title}` : ""}
              {job.duration ? ` · ${formatTime(job.duration)}` : ""}
            </p>
          )}
          <section className="panel log-panel">
            <h2>PROCESS LOG</h2>
            <div className="proc-log tall" ref={working ? undefined : logRef}>
              {logs.map((line) => (
                <div key={line}>{line}</div>
              ))}
            </div>
          </section>
        </aside>
      </section>

      {error && <p className="error">ERR // {error}</p>}

      <section className="panel results">
        <div className="panel-head">
          <div>
            <h2>03 // GENERATE</h2>
            <p className="muted">
              {clips.length
                ? `AI lock ${clips.length} momen. Generate jadi file video.`
                : "AI belum memilih. Jalankan langkah 02 dulu."}
            </p>
          </div>
          <button className="primary" disabled={working || clips.length === 0} onClick={() => void generateVideos()}>
            GENERATE {selected.length || clips.length} VIDEO
          </button>
        </div>

        <div className="clip-grid">
          {clips.map((clip, index) => (
            <article
              key={clip.id}
              className={`card ${activeId === clip.id ? "active" : ""}`}
              onClick={() => playClip(clip)}
            >
              <div className="badge">#{String(index + 1).padStart(2, "0")}</div>
              <h3>{clip.title}</h3>
              <p>{clip.reason}</p>
              <div className="meta">
                <span>
                  {formatTime(clip.start)}–{formatTime(clip.end)}
                </span>
                <span>{clipDuration(clip).toFixed(0)}s</span>
              </div>
            </article>
          ))}
        </div>

        {job?.exports?.length ? (
          <div className="downloads">
            <h3>04 // DOWNLOAD</h3>
            <div className="dl-list">
              {job.exports.map((item) => (
                <a key={item.url} href={item.url} download>
                  {item.name}
                </a>
              ))}
            </div>
          </div>
        ) : null}
      </section>
    </div>
  );
}
