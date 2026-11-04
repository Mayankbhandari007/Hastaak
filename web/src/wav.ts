/** Convert a browser recording (WebM/Opus from MediaRecorder) to 16 kHz mono 16-bit WAV, the format Muse Voice
 *  Transcribe accepts. Decoding and resampling are done by the browser's audio engine. */
export async function toWav16k(recording: Blob): Promise<Blob> {
  const RATE = 16000;
  const ctx = new AudioContext();
  try {
    const decoded = await ctx.decodeAudioData(await recording.arrayBuffer());
    const offline = new OfflineAudioContext(1, Math.max(1, Math.ceil(decoded.duration * RATE)), RATE);
    const src = offline.createBufferSource();
    src.buffer = decoded;                       // stereo is mixed down to the mono destination
    src.connect(offline.destination);
    src.start();
    return encodeWav((await offline.startRendering()).getChannelData(0), RATE);
  } finally {
    void ctx.close();
  }
}

function encodeWav(samples: Float32Array, rate: number): Blob {
  const view = new DataView(new ArrayBuffer(44 + samples.length * 2));
  const text = (at: number, s: string) => [...s].forEach((c, i) => view.setUint8(at + i, c.charCodeAt(0)));
  text(0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  text(8, "WAVE");
  text(12, "fmt ");
  view.setUint32(16, 16, true);                // PCM header size
  view.setUint16(20, 1, true);                 // PCM
  view.setUint16(22, 1, true);                 // mono
  view.setUint32(24, rate, true);
  view.setUint32(28, rate * 2, true);          // bytes per second
  view.setUint16(32, 2, true);                 // bytes per sample
  view.setUint16(34, 16, true);                // bits per sample
  text(36, "data");
  view.setUint32(40, samples.length * 2, true);
  samples.forEach((s, i) => view.setInt16(44 + i * 2, Math.max(-1, Math.min(1, s)) * 0x7fff, true));
  return new Blob([view], { type: "audio/wav" });
}
