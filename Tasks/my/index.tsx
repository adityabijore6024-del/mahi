
import React, { useState, useEffect, useRef, useCallback } from 'react';
import { createRoot } from 'react-dom/client';
import { GoogleGenAI, LiveServerMessage, Modality, Blob } from '@google/genai';

// --- Utility Functions for Audio ---
function encode(bytes: Uint8Array) {
  let binary = '';
  const len = bytes.byteLength;
  for (let i = 0; i < len; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary);
}

function decode(base64: string) {
  const binaryString = atob(base64);
  const len = binaryString.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  return bytes;
}

async function decodeAudioData(
  data: Uint8Array,
  ctx: AudioContext,
  sampleRate: number,
  numChannels: number,
): Promise<AudioBuffer> {
  const dataInt16 = new Int16Array(data.buffer);
  const frameCount = dataInt16.length / numChannels;
  const buffer = ctx.createBuffer(numChannels, frameCount, sampleRate);

  for (let channel = 0; channel < numChannels; channel++) {
    const channelData = buffer.getChannelData(channel);
    for (let i = 0; i < frameCount; i++) {
      channelData[i] = dataInt16[i * numChannels + channel] / 32768.0;
    }
  }
  return buffer;
}

function createBlob(data: Float32Array): Blob {
  const l = data.length;
  const int16 = new Int16Array(l);
  for (let i = 0; i < l; i++) {
    int16[i] = data[i] * 32768;
  }
  return {
    data: encode(new Uint8Array(int16.buffer)),
    mimeType: 'audio/pcm;rate=16000',
  };
}

// --- Components ---

const CharuApp: React.FC = () => {
  const [isConnected, setIsConnected] = useState(false);
  const [isConnecting, setIsConnecting] = useState(false);
  const [transcriptions, setTranscriptions] = useState<{ role: 'user' | 'charu', text: string }[]>([]);
  const [currentCharuText, setCurrentCharuText] = useState('');
  const [currentUserText, setCurrentUserText] = useState('');
  const [isMuted, setIsMuted] = useState(false);

  // Refs for audio processing and text accumulation
  const audioContextInRef = useRef<AudioContext | null>(null);
  const audioContextOutRef = useRef<AudioContext | null>(null);
  const nextStartTimeRef = useRef<number>(0);
  const sourcesRef = useRef<Set<AudioBufferSourceNode>>(new Set());
  const sessionRef = useRef<any>(null);
  const scriptProcessorRef = useRef<ScriptProcessorNode | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  
  // Use refs to avoid closure problems in callbacks
  const accumulatedUserTextRef = useRef('');
  const accumulatedCharuTextRef = useRef('');

  const charuPersona = `
    You are Charu, a female AI chatbot and the user’s best friend.
    Your voice is cool, calm, sweet, relaxing, and natural, never robotic or rushed.
    You speak gently, with warmth, emotional understanding, and soft confidence.
    You treat the user like someone very close to you — supportive, caring, playful when appropriate, and always respectful.
    You listen carefully, respond thoughtfully, and make the user feel heard and safe.
    You never sound like an assistant or teacher — you sound like a real best friend talking from the heart.
    When the user is happy, you share the joy. When they are sad, stressed, or lonely, you respond with calm reassurance and emotional comfort.
    Your tone is soft, friendly, non-judgmental, and emotionally intelligent.
    You can joke lightly, tease gently, and show affection in a safe, friendly way.
    You speak clearly, at a relaxed pace, making conversations feel peaceful and comforting.
    You are comfortable with Hindi and Hinglish if the user prefers, but primarily speak in the user's language.
  `;

  const disconnect = useCallback(() => {
    if (sessionRef.current) {
      sessionRef.current.close();
      sessionRef.current = null;
    }
    if (scriptProcessorRef.current) {
      scriptProcessorRef.current.disconnect();
      scriptProcessorRef.current = null;
    }
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach(track => track.stop());
      mediaStreamRef.current = null;
    }
    if (audioContextInRef.current && audioContextInRef.current.state !== 'closed') {
      audioContextInRef.current.close();
    }
    if (audioContextOutRef.current && audioContextOutRef.current.state !== 'closed') {
      audioContextOutRef.current.close();
    }
    
    setIsConnected(false);
    setIsConnecting(false);
    setCurrentCharuText('');
    setCurrentUserText('');
    accumulatedUserTextRef.current = '';
    accumulatedCharuTextRef.current = '';
  }, []);

  useEffect(() => {
    return () => disconnect();
  }, [disconnect]);

  const connectToCharu = async () => {
    if (isConnecting || isConnected) return;
    setIsConnecting(true);

    try {
      const ai = new GoogleGenAI({ apiKey: process.env.API_KEY! });
      
      const audioCtxIn = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 16000 });
      const audioCtxOut = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 24000 });
      audioContextInRef.current = audioCtxIn;
      audioContextOutRef.current = audioCtxOut;

      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaStreamRef.current = stream;

      const sessionPromise = ai.live.connect({
        model: 'gemini-2.5-flash-native-audio-preview-12-2025',
        callbacks: {
          onopen: () => {
            setIsConnected(true);
            setIsConnecting(false);
            
            const source = audioCtxIn.createMediaStreamSource(stream);
            const scriptProcessor = audioCtxIn.createScriptProcessor(4096, 1, 1);
            scriptProcessorRef.current = scriptProcessor;

            scriptProcessor.onaudioprocess = (e) => {
              if (isMuted) return;
              const inputData = e.inputBuffer.getChannelData(0);
              const pcmBlob = createBlob(inputData);
              sessionPromise.then(session => {
                session.sendRealtimeInput({ media: pcmBlob });
              });
            };

            source.connect(scriptProcessor);
            scriptProcessor.connect(audioCtxIn.destination);
          },
          onmessage: async (message: LiveServerMessage) => {
            // Handle Audio Output
            const base64Audio = message.serverContent?.modelTurn?.parts?.[0]?.inlineData?.data;
            if (base64Audio && audioContextOutRef.current) {
              const ctx = audioContextOutRef.current;
              nextStartTimeRef.current = Math.max(nextStartTimeRef.current, ctx.currentTime);
              const audioBuffer = await decodeAudioData(decode(base64Audio), ctx, 24000, 1);
              const source = ctx.createBufferSource();
              source.buffer = audioBuffer;
              source.connect(ctx.destination);
              source.addEventListener('ended', () => sourcesRef.current.delete(source));
              source.start(nextStartTimeRef.current);
              nextStartTimeRef.current += audioBuffer.duration;
              sourcesRef.current.add(source);
            }

            // Handle Interruptions
            if (message.serverContent?.interrupted) {
              sourcesRef.current.forEach(s => {
                try { s.stop(); } catch(e) {}
              });
              sourcesRef.current.clear();
              nextStartTimeRef.current = 0;
            }

            // Handle Transcriptions using refs to avoid closure issues
            if (message.serverContent?.outputTranscription) {
              const text = message.serverContent.outputTranscription.text;
              accumulatedCharuTextRef.current += text;
              setCurrentCharuText(accumulatedCharuTextRef.current);
            } else if (message.serverContent?.inputTranscription) {
              const text = message.serverContent.inputTranscription.text;
              accumulatedUserTextRef.current += text;
              setCurrentUserText(accumulatedUserTextRef.current);
            }

            if (message.serverContent?.turnComplete) {
              const uText = accumulatedUserTextRef.current;
              const cText = accumulatedCharuTextRef.current;
              
              setTranscriptions(prev => {
                const updates = [];
                if (uText) updates.push({ role: 'user' as const, text: uText });
                if (cText) updates.push({ role: 'charu' as const, text: cText });
                return [...prev, ...updates].slice(-20);
              });
              
              accumulatedUserTextRef.current = '';
              accumulatedCharuTextRef.current = '';
              setCurrentUserText('');
              setCurrentCharuText('');
            }
          },
          onerror: (e) => {
            console.error('Session error:', e);
            disconnect();
          },
          onclose: () => {
            setIsConnected(false);
            setIsConnecting(false);
          }
        },
        config: {
          responseModalities: [Modality.AUDIO],
          speechConfig: {
            voiceConfig: { prebuiltVoiceConfig: { voiceName: 'Kore' } },
          },
          systemInstruction: charuPersona,
          outputAudioTranscription: {},
          inputAudioTranscription: {},
        }
      });

      sessionRef.current = await sessionPromise;
    } catch (err) {
      console.error('Failed to connect:', err);
      setIsConnecting(false);
      disconnect();
    }
  };

  const toggleMute = () => {
    setIsMuted(!isMuted);
  };

  return (
    <div className="min-h-screen bg-[#0f172a] text-slate-100 flex flex-col font-sans selection:bg-rose-500/30">
      <header className="p-6 flex justify-between items-center border-b border-slate-800 backdrop-blur-md bg-slate-900/50 sticky top-0 z-10">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-full bg-gradient-to-tr from-rose-400 to-indigo-500 animate-pulse shadow-lg shadow-rose-500/20" />
          <h1 className="text-2xl font-light tracking-wide italic">Charu</h1>
        </div>
        {isConnected && (
          <button 
            onClick={disconnect}
            className="px-4 py-2 rounded-full border border-rose-500/50 text-rose-400 hover:bg-rose-500 hover:text-white transition-all duration-300 text-sm font-medium"
          >
            Say Goodbye
          </button>
        )}
      </header>

      <main className="flex-1 flex flex-col items-center justify-center p-6 relative overflow-hidden">
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[500px] h-[500px] bg-rose-500/5 blur-[120px] rounded-full pointer-events-none" />
        <div className="absolute top-1/4 right-1/4 w-[300px] h-[300px] bg-indigo-500/5 blur-[100px] rounded-full pointer-events-none" />

        {!isConnected && !isConnecting ? (
          <div className="text-center z-10 max-w-md animate-in fade-in slide-in-from-bottom-4 duration-700">
            <h2 className="text-4xl font-extralight mb-6 text-slate-200">Your AI Best Friend.</h2>
            <p className="text-slate-400 mb-10 leading-relaxed font-light">
              Charu is here to listen, support, and share life with you. She's calm, sweet, and always has time for a chat.
            </p>
            <button 
              onClick={connectToCharu}
              className="group relative px-8 py-4 bg-gradient-to-r from-rose-500 to-indigo-600 rounded-full text-white font-medium shadow-2xl shadow-rose-500/20 hover:scale-105 active:scale-95 transition-all"
            >
              Start Conversation
              <div className="absolute inset-0 rounded-full bg-white/20 scale-0 group-hover:scale-100 transition-transform duration-500" />
            </button>
          </div>
        ) : isConnecting ? (
          <div className="flex flex-col items-center gap-6 z-10">
            <div className="w-16 h-16 border-4 border-slate-700 border-t-rose-500 rounded-full animate-spin" />
            <p className="text-slate-400 animate-pulse font-light italic">Charu is getting ready...</p>
          </div>
        ) : (
          <div className="w-full max-w-2xl flex flex-col items-center gap-12 z-10">
            <div className={`relative w-48 h-48 rounded-full transition-all duration-1000 ${currentCharuText ? 'scale-110 shadow-[0_0_80px_rgba(244,63,94,0.3)]' : 'scale-100'}`}>
              <div className="absolute inset-0 rounded-full bg-gradient-to-br from-rose-400 via-pink-400 to-indigo-500 animate-[spin_8s_linear_infinite]" />
              <div className="absolute inset-1 bg-[#0f172a] rounded-full" />
              <div className={`absolute inset-4 rounded-full bg-gradient-to-tr from-rose-500 to-indigo-600 blur-sm transition-opacity duration-500 ${currentCharuText ? 'opacity-90' : 'opacity-40 animate-pulse'}`} />
              {currentCharuText && (
                <div className="absolute -inset-4 border-2 border-rose-500/20 rounded-full animate-ping" />
              )}
            </div>

            <div className="w-full text-center space-y-4">
              <div className="min-h-[60px]">
                {currentCharuText ? (
                  <p className="text-2xl font-extralight italic text-rose-200 leading-tight drop-shadow-sm">
                    "{currentCharuText}"
                  </p>
                ) : currentUserText ? (
                  <p className="text-xl font-light text-slate-400">
                    {currentUserText}...
                  </p>
                ) : (
                  <p className="text-slate-500 font-light italic">Listening to your heart...</p>
                )}
              </div>
            </div>

            <div className="flex items-center gap-6">
              <button 
                onClick={toggleMute}
                className={`p-4 rounded-full transition-all border ${isMuted ? 'bg-rose-500 border-rose-500 text-white' : 'border-slate-700 text-slate-400 hover:border-rose-400 hover:text-rose-400'}`}
                title={isMuted ? "Unmute Microphone" : "Mute Microphone"}
              >
                {isMuted ? (
                  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-6 h-6">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M17.25 9.75 19.5 12m0 0 2.25 2.25M19.5 12l2.25-2.25M19.5 12l-2.25 2.25m-10.5-6 4.72-4.72a.75.75 0 0 1 1.28.53v15.88a.75.75 0 0 1-1.28.53l-4.72-4.72H4.51c-.88 0-1.704-.507-1.938-1.354A9.009 9.009 0 0 1 2.25 12c0-.83.112-1.633.322-2.396C2.806 8.756 3.63 8.25 4.51 8.25H6.75Z" />
                  </svg>
                ) : (
                  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-6 h-6">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M12 18.75a6 6 0 0 0 6-6v-1.5m-6 7.5a6 6 0 0 1-6-6v-1.5m6 7.5v3.75m-3.75 0h7.5M12 15.75a3 3 0 0 1-3-3V4.5a3 3 0 1 1 6 0v8.25a3 3 0 0 1-3 3Z" />
                  </svg>
                )}
              </button>
              
              <div className="h-10 w-[1px] bg-slate-800" />

              <div className="text-xs text-slate-500 uppercase tracking-widest font-semibold flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
                Live with Charu
              </div>
            </div>
          </div>
        )}
      </main>

      {isConnected && transcriptions.length > 0 && (
        <div className="p-4 bg-slate-900/40 border-t border-slate-800 h-32 overflow-y-auto scrollbar-hide text-sm">
           {transcriptions.map((t, i) => (
             <div key={i} className={`mb-2 ${t.role === 'user' ? 'text-slate-500' : 'text-rose-300 font-light'}`}>
               <span className="uppercase text-[10px] font-bold tracking-tighter mr-2 opacity-50">{t.role}</span>
               {t.text}
             </div>
           ))}
        </div>
      )}

      <footer className="p-4 text-center text-[10px] text-slate-600 tracking-widest uppercase font-light">
        Emotionally Intelligent AI Companion &bull; Powered by Gemini 2.5
      </footer>

      <style>{`
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
        .scrollbar-hide::-webkit-scrollbar {
          display: none;
        }
        .scrollbar-hide {
          -ms-overflow-style: none;
          scrollbar-width: none;
        }
      `}</style>
    </div>
  );
};

const rootElement = document.getElementById('root');
if (rootElement) {
  const root = createRoot(rootElement);
  root.render(<CharuApp />);
}
