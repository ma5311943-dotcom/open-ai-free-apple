/**
 * OfflineAI - On-device AI assistant using llama.rn (llama.cpp).
 * Option 2: Modern Apple Light Theme with Lucide React Native Vector Icons.
 *
 * @format
 */

import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  ActivityIndicator,
  Alert,
  FlatList,
  Image as RNImage,
  Keyboard,
  KeyboardAvoidingView,
  Modal,
  NativeEventEmitter,
  NativeModules,
  PermissionsAndroid,
  Platform,
  Pressable,
  ScrollView,
  Share,
  StatusBar,
  StyleSheet,
  Switch,
  Text,
  TextInput,
  View,
} from 'react-native';
import RNFS from 'react-native-fs';
import { initLlama, LlamaContext } from 'llama.rn';
import {
  Brain,
  Camera,
  Check,
  ChevronDown,
  ChevronRight,
  Code2,
  Copy,
  Cpu,
  Download,
  FileText,
  FileUp,
  HardDrive,
  Image as ImageIcon,
  Lightbulb,
  Mail,
  Maximize2,
  Menu,
  MessageSquare,
  Mic,
  MicOff,
  Plus,
  Send,
  Settings,
  Sparkles,
  Square,
  Trash2,
  Volume2,
  VolumeX,
  X,
  Zap,
} from 'lucide-react-native';
import { launchCamera, launchImageLibrary } from 'react-native-image-picker';
import {
  pick,
  types,
  isErrorWithCode,
  errorCodes,
} from '@react-native-documents/picker';

// ---- Model configuration --------------------------------------------------
export type ModelCategory = 'text' | 'vision';

export type ModelOption = {
  id: string;
  category: ModelCategory;
  name: string;
  file: string;
  url: string;
  size: number;
  description: string;
  tag: string;
  mmprojFile?: string;
  mmprojUrl?: string;
  mmprojSize?: number;
};

export const AVAILABLE_MODELS: ModelOption[] = [
  // --- Text Reasoning Brains ---
  {
    id: 'qwen2.5-14b',
    category: 'text',
    name: 'Qwen2.5 14B Ultra Genius',
    file: 'Qwen2.5-14B-Instruct-Q4_K_M.gguf',
    url: 'https://huggingface.co/bartowski/Qwen2.5-14B-Instruct-GGUF/resolve/main/Qwen2.5-14B-Instruct-Q4_K_M.gguf',
    size: 8988110976,
    description: 'Ultra-intelligent 14B brain (Rivals GPT-4o-mini). Exceptional multi-step reasoning, advanced coding & all languages.',
    tag: 'Text Ultra · 9.0 GB',
  },
  {
    id: 'qwen2.5-7b',
    category: 'text',
    name: 'Qwen2.5 7B Super Genius',
    file: 'Qwen2.5-7B-Instruct-Q4_K_M.gguf',
    url: 'https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/resolve/main/Qwen2.5-7B-Instruct-Q4_K_M.gguf',
    size: 4683074240,
    description: 'High-intelligence 7B model. Outstanding logic, programming, mathematics & multilingual fluency.',
    tag: 'Text Genius · 4.7 GB',
  },
  {
    id: 'qwen2.5-3b',
    category: 'text',
    name: 'Qwen2.5 3B Smart',
    file: 'qwen2.5-3b-instruct-q4_k_m.gguf',
    url: 'https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf',
    size: 2104932768,
    description: 'Genius Multilingual AI (Urdu, English, Hindi, Arabic, 29+ languages, Fast & Smart).',
    tag: 'Text Smart · 2.1 GB',
  },
  {
    id: 'qwen2.5-1.5b',
    category: 'text',
    name: 'Qwen2.5 1.5B Fast',
    file: 'qwen2.5-1.5b-instruct-q4_k_m.gguf',
    url: 'https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf',
    size: 1117320736,
    description: 'Fast & Lightweight on-device assistant. Instant responses, minimal battery & RAM usage.',
    tag: 'Text Fast · 1.1 GB',
  },

  // --- Vision & Multimodal Brains ---
  {
    id: 'qwen2-vl-7b',
    category: 'vision',
    name: 'Qwen2-VL 7B Vision Master',
    file: 'Qwen2-VL-7B-Instruct-Q4_K_M.gguf',
    url: 'https://huggingface.co/bartowski/Qwen2-VL-7B-Instruct-GGUF/resolve/main/Qwen2-VL-7B-Instruct-Q4_K_M.gguf',
    size: 4683072672,
    mmprojFile: 'mmproj-Qwen2-VL-7B-Instruct-f16.gguf',
    mmprojUrl: 'https://huggingface.co/bartowski/Qwen2-VL-7B-Instruct-GGUF/resolve/main/mmproj-Qwen2-VL-7B-Instruct-f16.gguf',
    mmprojSize: 1352635904,
    description: 'Top-tier offline Vision AI. Directly sees WhatsApp screens, UI layouts, diagrams & real photos with high visual precision.',
    tag: 'Vision Master · 6.0 GB',
  },
  {
    id: 'qwen2-vl-2b',
    category: 'vision',
    name: 'Qwen2-VL 2B Vision Fast',
    file: 'Qwen2-VL-2B-Instruct-Q4_K_M.gguf',
    url: 'https://huggingface.co/bartowski/Qwen2-VL-2B-Instruct-GGUF/resolve/main/Qwen2-VL-2B-Instruct-Q4_K_M.gguf',
    size: 986047232,
    mmprojFile: 'mmproj-Qwen2-VL-2B-Instruct-f16.gguf',
    mmprojUrl: 'https://huggingface.co/bartowski/Qwen2-VL-2B-Instruct-GGUF/resolve/main/mmproj-Qwen2-VL-2B-Instruct-f16.gguf',
    mmprojSize: 1331656192,
    description: 'High-speed offline Vision AI. Light RAM footprint, instant pixel visual analysis for screenshots & camera photos.',
    tag: 'Vision Fast · 2.3 GB',
  },
];

const SETTINGS_PATH = `${RNFS.DocumentDirectoryPath}/settings.json`;
const SESSIONS_PATH = `${RNFS.DocumentDirectoryPath}/chat_sessions.json`;

const SYSTEM_PROMPT =
  'You are a helpful, concise AI assistant running on-device. ' +
  'Answer clearly and accurately in any language requested (English, Urdu, Hindi, Arabic, etc.). ' +
  'When an image or screenshot is attached, accurately inspect all visual details, identify the application interface (e.g. WhatsApp, Telegram, Android/iOS System, or Camera Photo), read all visible text, and answer questions directly without guessing or hallucinating.';
const STOP_WORDS = ['<|im_end|>', '<|endoftext|>', '</s>', '<|eot_id|>'];
const OCR_API_KEYS = ['helloworld', 'K88796853688957'];

type Phase = 'checking' | 'downloading' | 'loading' | 'ready' | 'error';
type Role = 'user' | 'assistant';
type Attachment = {
  id: string;
  type: 'image' | 'document';
  uri: string;
  name: string;
  size?: number;
  extractedText?: string;
  isExtracting?: boolean;
};
type Message = {
  id: string;
  role: Role;
  content: string;
  attachment?: Attachment;
};

export type ChatSession = {
  id: string;
  title: string;
  createdAt: number;
  updatedAt: number;
  messages: Message[];
};

// Helper to guarantee images are readable local files for llama.rn native C++ engine
async function ensureLocalFilePath(uri: string): Promise<string> {
  if (!uri) return '';
  if (Platform.OS === 'android' && uri.startsWith('content://')) {
    try {
      const dest = `${RNFS.CachesDirectoryPath}/vision_${Date.now()}_${Math.floor(Math.random() * 10000)}.jpg`;
      await RNFS.copyFile(uri, dest);
      return `file://${dest}`;
    } catch {
      try {
        const dest = `${RNFS.CachesDirectoryPath}/vision_${Date.now()}_${Math.floor(Math.random() * 10000)}.jpg`;
        const b64 = await RNFS.readFile(uri, 'base64');
        await RNFS.writeFile(dest, b64, 'base64');
        return `file://${dest}`;
      } catch (err) {
        console.warn('Could not cache content URI:', err);
      }
    }
  }
  return uri;
}

// ---- Hybrid Offline/Online OCR Engine -------------------------------------
async function extractTextFromImage(uri: string): Promise<string> {
  // 1. Try Offline Apple Vision / Native OCR module
  try {
    const { NativeOCR } = NativeModules;
    if (NativeOCR && typeof NativeOCR.recognizeText === 'function') {
      const nativeText = await NativeOCR.recognizeText(uri);
      if (nativeText && typeof nativeText === 'string' && nativeText.trim().length > 0) {
        return nativeText.trim();
      }
    }
  } catch {
    // Native module not linked or failed, fallback to online OCR
  }

  // 2. Online High-Accuracy OCR fallback via OCR.space API with dual keys
  try {
    let cleanUri = uri;
    if (Platform.OS === 'ios' && cleanUri.startsWith('file://')) {
      cleanUri = cleanUri.replace('file://', '');
    }
    const base64Data = await RNFS.readFile(cleanUri, 'base64');
    const base64Image = `data:image/jpeg;base64,${base64Data}`;

    for (const key of OCR_API_KEYS) {
      try {
        const params = new URLSearchParams({
          apikey: key,
          base64Image: base64Image,
          language: 'eng',
          isOverlayRequired: 'false',
          detectOrientation: 'true',
          scale: 'true',
        });

        const res = await fetch('https://api.ocr.space/parse/image', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
          },
          body: params.toString(),
        });

        const json = await res.json();
        if (json?.ParsedResults && json.ParsedResults.length > 0) {
          const parsed = json.ParsedResults[0]?.ParsedText;
          if (parsed && typeof parsed === 'string' && parsed.trim().length > 0) {
            return parsed.trim();
          }
        }
      } catch {
        // try next key
      }
    }
  } catch {
    // OCR fetch failed
  }

  return '';
}

// ---- Offline PDF Text Extraction Engine -----------------------------------
async function extractTextFromPdf(uri: string): Promise<string> {
  // 1. Try Native Apple PDFKit & Vision OCR / Android PDF extractor
  try {
    const { NativeOCR } = NativeModules;
    if (NativeOCR && typeof NativeOCR.extractPdfText === 'function') {
      const nativeText = await NativeOCR.extractPdfText(uri);
      if (nativeText && typeof nativeText === 'string' && nativeText.trim().length > 0) {
        return nativeText.trim();
      }
    }
  } catch {
    // Native module not linked or failed
  }

  // 2. Offline fallback: read raw file and extract text strings
  try {
    let cleanUri = uri;
    if (Platform.OS === 'ios' && cleanUri.startsWith('file://')) {
      cleanUri = cleanUri.replace('file://', '');
    }
    const raw = await RNFS.readFile(cleanUri, 'ascii');
    // Match uncompressed PDF text operators: (text) Tj or [(t)(e)(x)(t)] TJ
    const matches = raw.match(/\((.*?)\)\s*Tj/g) || raw.match(/\[(.*?)\]\s*TJ/g);
    if (matches && matches.length > 0) {
      const text = matches
        .map(m => m.replace(/[\(\)\[\]]|Tj|TJ/g, '').trim())
        .filter(Boolean)
        .join(' ');
      if (text.length > 20) return text;
    }

    // Extract readable ascii strings of length >= 5
    const textBlocks = raw.match(/[A-Za-z0-9\s.,!?:;'"()\/\-]{5,}/g);
    if (textBlocks && textBlocks.length > 0) {
      const filtered = textBlocks.filter(
        b =>
          !b.includes('/Filter') &&
          !b.includes('/Length') &&
          !b.includes('/Type') &&
          !b.includes('/Font') &&
          !b.includes('/MediaBox')
      );
      if (filtered.length > 0) {
        return filtered.join('\n').substring(0, 30000);
      }
    }
  } catch {
    // raw read failed
  }

  return '';
}

// ---- Design tokens (Option 2: Modern Apple Light Theme) -------------------
const C = {
  bg: '#F3F4F6', // Soft crisp platinum canvas (Expo Go style)
  surface: '#FFFFFF', // Pure white cards & navigation
  surface2: '#F9FAFB', // Off-white for secondary cards
  border: '#E5E7EB', // Subtle Apple gray border
  borderDark: '#D1D5DB', // Input field border
  text: '#111827', // Deep charcoal/black
  textSecondary: '#4B5563', // Secondary body text
  muted: '#9CA3AF', // Neutral gray
  accent: '#4F46E5', // Vibrant Indigo/Royal Blue
  accentSoft: '#EEF2FF', // Tinted indigo for chips
  accentGreen: '#10B981', // Mint green dot
  accentGreenBg: '#ECFDF5', // Soft green badge background
  codeBg: '#181825', // Deep dark obsidian for code blocks
  codeText: '#CDD6F4', // Crisp monospaced code text
  codeBorder: '#27273A', // Dark code border
  codeHeader: '#1E1E2E', // Dark code header
  codeLang: '#89DCEB', // Cyan language label
  danger: '#EF4444', // Red
};

function formatSize(bytes: number) {
  if (bytes >= 1024 * 1024 * 1024) {
    return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`;
  }
  return `${(bytes / (1024 * 1024)).toFixed(0)} MB`;
}

function formatDetailedSize(bytes: number) {
  if (!bytes || bytes <= 0) return '0 MB';
  if (bytes >= 1024 * 1024 * 1024) {
    return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
  }
  if (bytes >= 1024 * 1024) {
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  }
  if (bytes >= 1024) {
    return `${(bytes / 1024).toFixed(0)} KB`;
  }
  return `${bytes} B`;
}

function formatMB(bytes: number) {
  return formatSize(bytes);
}

function formatMBOnly(bytes: number) {
  if (!bytes || bytes <= 0) return '0 MB';
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

// Clean markdown syntax for natural-sounding offline voice reading
function cleanMarkdownForSpeech(text: string): string {
  if (!text) return '';
  return text
    .replace(/```[\s\S]*?```/g, ' Code snippet omitted. ')
    .replace(/`([^`]+)`/g, '$1')
    .replace(/^#{1,6}\s+/gm, '')
    .replace(/(\*\*|__)(.*?)\1/g, '$2')
    .replace(/(\*|_)(.*?)\1/g, '$2')
    .replace(/~~(.*?)~~/g, '$1')
    .replace(/\[([^\]]+)\]\([^\)]+\)/g, '$1')
    .replace(/https?:\/\/\S+/gi, ' link ')
    .replace(/[|•▪▫★☆]/g, ' ')
    .replace(/^\s*>\s+/gm, '')
    .replace(/^\s*[-*+]\s+/gm, '')
    .replace(/^\s*\d+\.\s+/gm, '')
    .replace(/\s+/g, ' ')
    .trim();
}

// Clean and format user speech for maximum LLM comprehension and accuracy
function cleanSpokenPrompt(rawText: string): string {
  if (!rawText) return '';
  let text = rawText.trim();

  // Strip common hesitation sounds
  text = text.replace(/^(uh+|um+|ah+|er+|like)\s+/i, '');
  text = text.replace(/\s+(uh+|um+|ah+|er+)\s+/gi, ' ');

  // Capitalize first letter
  if (text.length > 0) {
    text = text.charAt(0).toUpperCase() + text.slice(1);
  }

  // Ensure question mark if question word starts the prompt
  const isQuestion = /^(what|how|why|when|where|who|which|whose|can|could|would|will|is|are|do|does|did|should)/i.test(
    text,
  );
  if (isQuestion && !/[?!.]$/.test(text)) {
    text += '?';
  }

  return text;
}

function App() {
  return (
    <View style={styles.root}>
      <StatusBar barStyle="dark-content" />
      <ChatScreen />
    </View>
  );
}

function ChatScreen() {
  const [selectedModelId, setSelectedModelId] = useState<string>('qwen2.5-1.5b');
  const [phase, setPhase] = useState<Phase>('checking');
  const [progress, setProgress] = useState(0);
  const [downloaded, setDownloaded] = useState(0);
  const [downloadSpeed, setDownloadSpeed] = useState<string>('');
  const [downloadEta, setDownloadEta] = useState<string>('');
  const [downloadTitle, setDownloadTitle] = useState<string>('');
  const [modelCategoryFilter, setModelCategoryFilter] = useState<ModelCategory>('text');
  const isMultimodalReadyRef = useRef(false);
  const [isMultimodalReady, setIsMultimodalReady] = useState(false);
  const speedCalcRef = useRef<{ lastBytes: number; lastTime: number }>({ lastBytes: 0, lastTime: 0 });
  const isDownloadingRef = useRef(false);
  const activeDownloadModelIdRef = useRef<string | null>(null);
  const selectedModelIdRef = useRef(selectedModelId);
  selectedModelIdRef.current = selectedModelId;
  const [error, setError] = useState('');
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [generating, setGenerating] = useState(false);
  const [speed, setSpeed] = useState('');

  const [systemPrompt, setSystemPrompt] = useState(SYSTEM_PROMPT);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [settingsTab, setSettingsTab] = useState<'models' | 'downloads' | 'persona'>('models');
  const [tempPrompt, setTempPrompt] = useState(SYSTEM_PROMPT);
  const [keyboardVisible, setKeyboardVisible] = useState(false);
  const [downloadMinimized, setDownloadMinimized] = useState(false);
  const [downloadTotal, setDownloadTotal] = useState(0);
  const [downloadingModelId, setDownloadingModelId] = useState<string | null>(null);
  const [activeTextModelId, setActiveTextModelId] = useState<string>('qwen2.5-1.5b');
  const [activeVisionModelId, setActiveVisionModelId] = useState<string>('qwen2-vl-7b');
  const [attachment, setAttachment] = useState<Attachment | null>(null);
  const [attachMenuOpen, setAttachMenuOpen] = useState(false);
  const isPickingMediaRef = useRef(false);

  // Model hub: track which models are downloaded on device storage
  const [downloadedModels, setDownloadedModels] = useState<Record<string, boolean>>({});
  const downloadJobIdRef = useRef<number | null>(null);

  // Sessions / Chat History state (matching Image 2)
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [currentSessionId, setCurrentSessionId] = useState<string>('');
  const [drawerOpen, setDrawerOpen] = useState(false);

  // Voice recording & Offline Speech state
  const [isRecording, setIsRecording] = useState(false);
  const [recordingSeconds, setRecordingSeconds] = useState(0);
  const recordingTimer = useRef<any>(null);
  const [voiceResponseEnabled, setVoiceResponseEnabled] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [speakingMsgId, setSpeakingMsgId] = useState<string | null>(null);
  const isSpeakingRef = useRef(false);
  isSpeakingRef.current = isSpeaking;
  const isRecordingRef = useRef(isRecording);
  isRecordingRef.current = isRecording;
  const toggleRecordingRef = useRef<() => Promise<void> | void>(() => {});
  const voiceResponseEnabledRef = useRef(voiceResponseEnabled);
  voiceResponseEnabledRef.current = voiceResponseEnabled;
  const silenceTimerRef = useRef<any>(null);

  // User scroll position control (prevents forced auto-scroll if user scrolled up)
  const isUserAtBottomRef = useRef(true);
  const isUserTouchingScrollRef = useRef(false);
  const [showScrollBottomBtn, setShowScrollBottomBtn] = useState(false);

  const ctxRef = useRef<LlamaContext | null>(null);
  const listRef = useRef<FlatList<Message>>(null);

  const activeModel =
    AVAILABLE_MODELS.find(m => m.id === selectedModelId) || AVAILABLE_MODELS[0];

  const activeDownloadModel =
    AVAILABLE_MODELS.find(m => m.id === downloadingModelId) ||
    AVAILABLE_MODELS.find(m => m.id === activeDownloadModelIdRef.current) ||
    activeModel;

  // Refresh which models are stored locally on the phone (verifies main file and mmproj if vision)
  const refreshDownloadedStatus = useCallback(async () => {
    const status: Record<string, boolean> = {};
    for (const m of AVAILABLE_MODELS) {
      try {
        const path = `${RNFS.DocumentDirectoryPath}/${m.file}`;
        let modelOk = false;
        if (await RNFS.exists(path)) {
          const stat = await RNFS.stat(path);
          modelOk = Math.abs(Number(stat.size) - m.size) <= 4096;
        }
        if (m.category === 'vision' && m.mmprojFile && m.mmprojSize) {
          const mmPath = `${RNFS.DocumentDirectoryPath}/${m.mmprojFile}`;
          let mmOk = false;
          if (await RNFS.exists(mmPath)) {
            const stat = await RNFS.stat(mmPath);
            mmOk = Math.abs(Number(stat.size) - m.mmprojSize) <= 4096;
          }
          status[m.id] = modelOk && mmOk;
        } else {
          status[m.id] = modelOk;
        }
      } catch {
        status[m.id] = false;
      }
    }
    setDownloadedModels(status);
  }, []);

  useEffect(() => {
    refreshDownloadedStatus();
    // Auto-clean any leftover .part files from previous interrupted downloads to free up device storage
    (async () => {
      try {
        for (const m of AVAILABLE_MODELS) {
          const tempPath = `${RNFS.DocumentDirectoryPath}/${m.file}.part`;
          if (await RNFS.exists(tempPath)) {
            await RNFS.unlink(tempPath);
          }
          if (m.mmprojFile) {
            const tempMmPath = `${RNFS.DocumentDirectoryPath}/${m.mmprojFile}.part`;
            if (await RNFS.exists(tempMmPath)) {
              await RNFS.unlink(tempMmPath);
            }
          }
        }
      } catch {}
    })();
  }, [refreshDownloadedStatus]);

  useEffect(() => {
    if (settingsOpen) {
      refreshDownloadedStatus();
    }
  }, [settingsOpen, refreshDownloadedStatus]);

  // Load chat sessions from disk on startup
  useEffect(() => {
    RNFS.exists(SESSIONS_PATH).then(async exists => {
      if (exists) {
        try {
          const raw = await RNFS.readFile(SESSIONS_PATH, 'utf8');
          const loaded: ChatSession[] = JSON.parse(raw);
          if (Array.isArray(loaded) && loaded.length > 0) {
            setSessions(loaded);
            const active = loaded[0];
            setCurrentSessionId(active.id);
            setMessages(active.messages || []);
            return;
          }
        } catch {}
      }
      const initialId = `${Date.now()}`;
      const defaultSession: ChatSession = {
        id: initialId,
        title: 'New conversation',
        createdAt: Date.now(),
        updatedAt: Date.now(),
        messages: [],
      };
      setSessions([defaultSession]);
      setCurrentSessionId(initialId);
      setMessages([]);
    });
  }, []);

  // Helper to persist updated messages to active session
  const persistSessionMessages = useCallback((updatedMsgs: Message[]) => {
    setSessions(prevSessions => {
      const activeId = currentSessionId || `${Date.now()}`;
      let found = false;
      const updated = prevSessions.map(s => {
        if (s.id === activeId) {
          found = true;
          let title = s.title;
          if (title === 'New conversation' && updatedMsgs.length > 0) {
            const firstUser = updatedMsgs.find(m => m.role === 'user');
            if (firstUser?.content?.trim()) {
              title = firstUser.content.trim().slice(0, 36);
            } else if (firstUser?.attachment) {
              title = `${firstUser.attachment.type === 'image' ? 'Photo' : 'Doc'}: ${firstUser.attachment.name.slice(0, 24)}`;
            }
          }
          return {
            ...s,
            title,
            updatedAt: Date.now(),
            messages: updatedMsgs,
          };
        }
        return s;
      });

      if (!found) {
        let title = 'New conversation';
        const firstUser = updatedMsgs.find(m => m.role === 'user');
        if (firstUser?.content?.trim()) {
          title = firstUser.content.trim().slice(0, 36);
        } else if (firstUser?.attachment) {
          title = `${firstUser.attachment.type === 'image' ? 'Photo' : 'Doc'}: ${firstUser.attachment.name.slice(0, 24)}`;
        }
        const newSession: ChatSession = {
          id: activeId,
          title,
          createdAt: Date.now(),
          updatedAt: Date.now(),
          messages: updatedMsgs,
        };
        updated.unshift(newSession);
      }

      RNFS.writeFile(SESSIONS_PATH, JSON.stringify(updated), 'utf8').catch(() => {});
      return updated;
    });
  }, [currentSessionId]);

  const startNewChat = useCallback(() => {
    Keyboard.dismiss();
    const newId = `${Date.now()}`;
    const newSession: ChatSession = {
      id: newId,
      title: 'New conversation',
      createdAt: Date.now(),
      updatedAt: Date.now(),
      messages: [],
    };
    setSessions(prev => {
      const filtered = prev.filter(s => s.messages.length > 0);
      const next = [newSession, ...filtered];
      RNFS.writeFile(SESSIONS_PATH, JSON.stringify(next), 'utf8').catch(() => {});
      return next;
    });
    setCurrentSessionId(newId);
    setMessages([]);
    setAttachment(null);
    setInput('');
    setDrawerOpen(false);
  }, []);

  const selectSession = useCallback((session: ChatSession) => {
    Keyboard.dismiss();
    setCurrentSessionId(session.id);
    setMessages(session.messages || []);
    setAttachment(null);
    setInput('');
    setDrawerOpen(false);
  }, []);

  const deleteSession = useCallback((sessionId: string, title: string) => {
    Alert.alert(
      'Delete Conversation?',
      `Are you sure you want to delete "${title || 'this chat'}"? This will permanently delete its messages and free storage.`,
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Delete',
          style: 'destructive',
          onPress: () => {
            setSessions(prev => {
              const filtered = prev.filter(s => s.id !== sessionId);
              let nextSessions = filtered;
              if (filtered.length === 0) {
                const freshId = `${Date.now()}`;
                nextSessions = [
                  {
                    id: freshId,
                    title: 'New conversation',
                    createdAt: Date.now(),
                    updatedAt: Date.now(),
                    messages: [],
                  },
                ];
              }
              RNFS.writeFile(SESSIONS_PATH, JSON.stringify(nextSessions), 'utf8').catch(() => {});

              if (sessionId === currentSessionId) {
                const target = nextSessions[0];
                setCurrentSessionId(target.id);
                setMessages(target.messages || []);
              }
              return nextSessions;
            });
          },
        },
      ],
    );
  }, [currentSessionId]);

  const clearAllSessions = useCallback(() => {
    Alert.alert(
      'Clear All Chat History?',
      'This will permanently delete all conversation history and free up storage on your phone. This action cannot be undone.',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Clear All',
          style: 'destructive',
          onPress: async () => {
            const freshId = `${Date.now()}`;
            const freshSession: ChatSession = {
              id: freshId,
              title: 'New conversation',
              createdAt: Date.now(),
              updatedAt: Date.now(),
              messages: [],
            };
            setSessions([freshSession]);
            setCurrentSessionId(freshId);
            setMessages([]);
            setAttachment(null);
            setInput('');
            setDrawerOpen(false);
            try {
              await RNFS.writeFile(SESSIONS_PATH, JSON.stringify([freshSession]), 'utf8');
              Alert.alert('History Cleared', 'All chat sessions deleted and storage freed.');
            } catch {}
          },
        },
      ],
    );
  }, []);

  // Monitor keyboard visibility for modal adjustment
  useEffect(() => {
    const showSub = Keyboard.addListener(
      Platform.OS === 'ios' ? 'keyboardWillShow' : 'keyboardDidShow',
      () => setKeyboardVisible(true),
    );
    const hideSub = Keyboard.addListener(
      Platform.OS === 'ios' ? 'keyboardWillHide' : 'keyboardDidHide',
      () => setKeyboardVisible(false),
    );
    return () => {
      showSub.remove();
      hideSub.remove();
    };
  }, []);

  // Cancel in-progress download
  const cancelDownload = useCallback(async () => {
    isDownloadingRef.current = false;
    activeDownloadModelIdRef.current = null;
    setDownloadingModelId(null);
    setDownloadMinimized(false);
    if (downloadJobIdRef.current !== null) {
      try {
        RNFS.stopDownload(downloadJobIdRef.current);
      } catch {}
      downloadJobIdRef.current = null;
    }
    const { DownloadNotificationModule } = NativeModules;
    if (Platform.OS === 'android' && DownloadNotificationModule?.stopForegroundService) {
      try {
        await DownloadNotificationModule.stopForegroundService();
      } catch {}
    }
    const model = activeDownloadModel;
    try {
      const tempPath = `${RNFS.DocumentDirectoryPath}/${model.file}.part`;
      if (await RNFS.exists(tempPath)) {
        await RNFS.unlink(tempPath);
      }
      if (model.mmprojFile) {
        const tempMm = `${RNFS.DocumentDirectoryPath}/${model.mmprojFile}.part`;
        if (await RNFS.exists(tempMm)) {
          await RNFS.unlink(tempMm);
        }
      }
    } catch {}
    setProgress(0);
    setDownloaded(0);
    setDownloadTotal(0);
    setDownloadSpeed('');
    setDownloadEta('');
    setDownloadTitle('');
    setPhase('error');
    setError('Download was canceled.');
  }, [activeDownloadModel]);

  // Model preparation and switching (Protected against duplicate concurrent triggers)
  const prepareModel = useCallback(async (targetModelId?: string) => {
    const { DownloadNotificationModule } = NativeModules;
    const modelId = targetModelId || selectedModelIdRef.current;
    const model =
      AVAILABLE_MODELS.find(m => m.id === modelId) || AVAILABLE_MODELS[0];
    const modelPath = `${RNFS.DocumentDirectoryPath}/${model.file}`;
    const tempPath = `${modelPath}.part`;

    const isVision = model.category === 'vision' && !!model.mmprojFile && !!model.mmprojUrl;
    const mmprojPath = isVision ? `${RNFS.DocumentDirectoryPath}/${model.mmprojFile}` : '';
    const tempMmprojPath = isVision ? `${mmprojPath}.part` : '';

    // If this exact model is ALREADY downloading, do NOT restart or unlink!
    if (isDownloadingRef.current && activeDownloadModelIdRef.current === model.id) {
      return;
    }

    // If another download was running, cancel it before switching
    if (downloadJobIdRef.current !== null && activeDownloadModelIdRef.current !== model.id) {
      try {
        RNFS.stopDownload(downloadJobIdRef.current);
      } catch {}
      downloadJobIdRef.current = null;
      if (Platform.OS === 'android' && DownloadNotificationModule?.stopForegroundService) {
        try {
          await DownloadNotificationModule.stopForegroundService();
        } catch {}
      }
    }

    try {
      setError('');
      setPhase('checking');

      let modelReady = false;
      if (await RNFS.exists(modelPath)) {
        const stat = await RNFS.stat(modelPath);
        modelReady = Math.abs(Number(stat.size) - model.size) <= 4096;
        if (!modelReady) {
          await RNFS.unlink(modelPath);
        }
      }

      let mmprojReady = !isVision;
      if (isVision && (await RNFS.exists(mmprojPath))) {
        const stat = await RNFS.stat(mmprojPath);
        mmprojReady = Math.abs(Number(stat.size) - (model.mmprojSize || 0)) <= 4096;
        if (!mmprojReady) {
          await RNFS.unlink(mmprojPath);
        }
      }

      const ready = modelReady && mmprojReady;

      if (!ready) {
        setPhase('downloading');
        isDownloadingRef.current = true;
        activeDownloadModelIdRef.current = model.id;
        setDownloadingModelId(model.id);
        setDownloadTotal(model.size);

        // On Android 13+, request notification permission
        if (Platform.OS === 'android' && Platform.Version >= 33) {
          try {
            await PermissionsAndroid.request(
              PermissionsAndroid.PERMISSIONS.POST_NOTIFICATIONS,
            );
          } catch {}
        }

        const downloadSingleFile = async (
          fromUrl: string,
          toTemp: string,
          toDest: string,
          expectedSize: number,
          label: string,
        ) => {
          setDownloadTitle(label);
          setDownloadTotal(expectedSize);
          setProgress(0);
          setDownloaded(0);
          setDownloadSpeed('');
          setDownloadEta('');
          speedCalcRef.current = { lastBytes: 0, lastTime: Date.now() };

          if (await RNFS.exists(toTemp)) {
            await RNFS.unlink(toTemp);
          }

          if (Platform.OS === 'android' && DownloadNotificationModule?.startForegroundService) {
            try {
              await DownloadNotificationModule.startForegroundService(
                label,
                '',
                'Starting download...',
                '0 MB',
                formatSize(expectedSize),
                formatSize(expectedSize),
                0,
              );
            } catch (err) {
              console.warn('Could not start download foreground service:', err);
            }
          }

          let lastNotifyTime = 0;
          let currentSpeedStr = '';
          let currentEtaStr = '';

          const job = RNFS.downloadFile({
            fromUrl,
            toFile: toTemp,
            background: true,
            discretionary: false,
            progressInterval: 250,
            begin: () => {
              speedCalcRef.current = { lastBytes: 0, lastTime: Date.now() };
            },
            progress: res => {
              const total =
                res.contentLength && res.contentLength > 100000000
                  ? res.contentLength
                  : expectedSize;
              const written = res.bytesWritten;
              setDownloaded(written);
              setDownloadTotal(total);
              setProgress(Math.min(1, Math.max(0, written / total)));

              const now = Date.now();
              const timeDelta = (now - speedCalcRef.current.lastTime) / 1000;
              if (timeDelta >= 1) {
                const bytesDelta = written - speedCalcRef.current.lastBytes;
                if (bytesDelta > 0) {
                  const spd = bytesDelta / timeDelta;
                  currentSpeedStr = `${(spd / (1024 * 1024)).toFixed(1)} MB/s`;
                  setDownloadSpeed(currentSpeedStr);
                  const remaining = Math.max(0, total - written);
                  const secs = Math.round(remaining / spd);
                  currentEtaStr = secs > 60 ? `~${Math.ceil(secs / 60)} min left` : `~${secs}s left`;
                  setDownloadEta(currentEtaStr);
                }
                speedCalcRef.current = { lastBytes: written, lastTime: now };
              }

              if (now - lastNotifyTime >= 1000) {
                lastNotifyTime = now;
                const percent = Math.min(100, Math.max(0, Math.round((written / total) * 100)));
                const leftBytes = Math.max(0, total - written);

                if (Platform.OS === 'android' && DownloadNotificationModule?.updateProgress) {
                  DownloadNotificationModule.updateProgress(
                    label,
                    currentSpeedStr,
                    currentEtaStr,
                    formatDetailedSize(written),
                    formatSize(total),
                    formatDetailedSize(leftBytes),
                    percent,
                  );
                }
              }
            },
          });

          downloadJobIdRef.current = job.jobId;
          const result = await job.promise;
          downloadJobIdRef.current = null;

          if (result.statusCode !== 200) {
            throw new Error(`Download failed (HTTP ${result.statusCode})`);
          }
          const stat = await RNFS.stat(toTemp);
          if (Math.abs(Number(stat.size) - expectedSize) > 4096) {
            throw new Error('Download incomplete. Check internet connection and retry.');
          }
          await RNFS.moveFile(toTemp, toDest);
        };

        // Step 1: Download main model weights if not ready
        if (!modelReady) {
          const label = isVision ? `Downloading ${model.name} (Brain 1/2)` : `Downloading ${model.name}`;
          await downloadSingleFile(model.url, tempPath, modelPath, model.size, label);
        }

        // Step 2: Download vision mmproj eyes if vision and not ready
        if (isVision && !mmprojReady && model.mmprojUrl && model.mmprojSize) {
          const label = `Downloading ${model.name} (Vision Eyes 2/2)`;
          await downloadSingleFile(model.mmprojUrl, tempMmprojPath, mmprojPath, model.mmprojSize, label);
        }

        isDownloadingRef.current = false;
        activeDownloadModelIdRef.current = null;
        setDownloadingModelId(null);
        setDownloadTitle('');
        setDownloadMinimized(false);

        if (Platform.OS === 'android' && DownloadNotificationModule?.stopForegroundService) {
          try {
            await DownloadNotificationModule.stopForegroundService();
          } catch {}
        }

        await refreshDownloadedStatus();
      }

      setPhase('loading');
      if (ctxRef.current) {
        try {
          ctxRef.current.release();
        } catch {}
        ctxRef.current = null;
      }

      // Context size: optimize for 14B / 7B or vision models to avoid RAM exhaustion
      const isLargeModel =
        model.id === 'qwen2.5-14b' ||
        model.id === 'qwen2-vl-7b' ||
        model.size > 4000000000;
      const ctx = await initLlama({
        model: `file://${modelPath}`,
        n_ctx: isLargeModel ? 2048 : 4096,
        n_gpu_layers: 0,
        use_mlock: false,
        // Disable context shifting for multimodal vision models to preserve image token alignment
        ctx_shift: !isVision,
      });

      // Initialize multimodal vision support if vision model
      if (isVision && model.mmprojFile) {
        try {
          const ok = await ctx.initMultimodal({
            path: `file://${mmprojPath}`,
            use_gpu: false,
            image_min_tokens: 256,
            image_max_tokens: 1024,
          });
          isMultimodalReadyRef.current = ok;
          setIsMultimodalReady(ok);
        } catch (mErr) {
          console.warn('Multimodal vision initialization error:', mErr);
          isMultimodalReadyRef.current = false;
          setIsMultimodalReady(false);
        }
      } else {
        isMultimodalReadyRef.current = false;
        setIsMultimodalReady(false);
      }

      ctxRef.current = ctx;
      setSelectedModelId(model.id);
      setPhase('ready');
      await refreshDownloadedStatus();
    } catch (e: any) {
      if (Platform.OS === 'android' && DownloadNotificationModule?.stopForegroundService) {
        try {
          await DownloadNotificationModule.stopForegroundService();
        } catch {}
      }
      isDownloadingRef.current = false;
      activeDownloadModelIdRef.current = null;
      setDownloadingModelId(null);
      setDownloadMinimized(false);
      downloadJobIdRef.current = null;
      setDownloadTitle('');
      setError(e?.message ?? String(e));
      setPhase('error');
    }
  }, [refreshDownloadedStatus]);

  // Load saved personalization and saved model choice on mount
  useEffect(() => {
    let isMounted = true;
    (async () => {
      try {
        let modelToLoad = 'qwen2.5-1.5b';
        if (await RNFS.exists(SETTINGS_PATH)) {
          const raw = await RNFS.readFile(SETTINGS_PATH, 'utf8');
          const data = JSON.parse(raw);
          if (data.systemPrompt) {
            setSystemPrompt(data.systemPrompt);
            setTempPrompt(data.systemPrompt);
          }
          if (data.selectedModelId) {
            modelToLoad = data.selectedModelId;
          }
          if (typeof data.voiceResponseEnabled === 'boolean') {
            setVoiceResponseEnabled(data.voiceResponseEnabled);
          }
          if (data.activeTextModelId) {
            setActiveTextModelId(data.activeTextModelId);
          }
          if (data.activeVisionModelId) {
            setActiveVisionModelId(data.activeVisionModelId);
          }
        }
        if (isMounted) {
          setSelectedModelId(modelToLoad);
          prepareModel(modelToLoad);
        }
      } catch {
        if (isMounted) {
          prepareModel('qwen2.5-1.5b');
        }
      }
    })();
    return () => {
      isMounted = false;
      ctxRef.current?.release();
      ctxRef.current = null;
    };
  }, [prepareModel]);

  // Listen for native TTS events to sync speaking state
  // Listen for native TTS and speech events to sync speaking state and auto-silence
  useEffect(() => {
    const { NativeSpeech } = NativeModules;
    if (!NativeSpeech) return;

    let subStart: any;
    let subDone: any;
    let subError: any;
    let subPartial: any;

    try {
      const emitter = new NativeEventEmitter(NativeSpeech);
      subStart = emitter.addListener('onTtsStart', () => {
        setIsSpeaking(true);
      });
      subDone = emitter.addListener('onTtsDone', () => {
        setIsSpeaking(false);
        setSpeakingMsgId(null);
      });
      subError = emitter.addListener('onTtsError', () => {
        setIsSpeaking(false);
        setSpeakingMsgId(null);
      });
      // Smart Auto-Silence: after speaking, 1.4s of silence auto-finalizes and sends
      subPartial = emitter.addListener('onSpeechPartialResults', (data: any) => {
        const text = data?.text?.trim();
        if (text && text.length > 0) {
          if (silenceTimerRef.current) {
            clearTimeout(silenceTimerRef.current);
          }
          if (voiceResponseEnabledRef.current && isRecordingRef.current) {
            silenceTimerRef.current = setTimeout(() => {
              if (isRecordingRef.current) {
                toggleRecordingRef.current();
              }
            }, 1400);
          }
        }
      });
    } catch {}

    return () => {
      subStart?.remove?.();
      subDone?.remove?.();
      subError?.remove?.();
      subPartial?.remove?.();
      if (silenceTimerRef.current) {
        clearTimeout(silenceTimerRef.current);
      }
      try {
        NativeSpeech.stopSpeaking?.();
      } catch {}
    };
  }, []);

  const stopSpeaking = useCallback(async () => {
    try {
      const { NativeSpeech } = NativeModules;
      if (NativeSpeech && typeof NativeSpeech.stopSpeaking === 'function') {
        await NativeSpeech.stopSpeaking();
      }
    } catch {}
    setIsSpeaking(false);
    setSpeakingMsgId(null);
  }, []);

  const speakText = useCallback(
    async (text: string, msgId?: string) => {
      const clean = cleanMarkdownForSpeech(text);
      if (!clean) return;

      try {
        const { NativeSpeech } = NativeModules;
        if (NativeSpeech && typeof NativeSpeech.speak === 'function') {
          if (isSpeakingRef.current && speakingMsgId === msgId) {
            await stopSpeaking();
            return;
          }
          setIsSpeaking(true);
          setSpeakingMsgId(msgId || null);
          await NativeSpeech.speak(clean, { queueMode: 'flush', rate: 1.0, pitch: 1.0 });
        }
      } catch {
        setIsSpeaking(false);
        setSpeakingMsgId(null);
      }
    },
    [speakingMsgId, stopSpeaking],
  );

  // Zero-latency sentence streamer: speaks first sentence immediately, queues subsequent ones
  const speakSentence = useCallback(
    async (sentence: string, isFirst: boolean, msgId?: string) => {
      const clean = cleanMarkdownForSpeech(sentence);
      if (!clean) return;

      try {
        const { NativeSpeech } = NativeModules;
        if (NativeSpeech && typeof NativeSpeech.speak === 'function') {
          setIsSpeaking(true);
          setSpeakingMsgId(msgId || null);
          await NativeSpeech.speak(clean, {
            queueMode: isFirst ? 'flush' : 'add',
            rate: 1.0,
            pitch: 1.0,
          });
        }
      } catch {
        setIsSpeaking(false);
        setSpeakingMsgId(null);
      }
    },
    [],
  );

  const speakMessage = useCallback(
    (msgId: string, content: string) => {
      if (isSpeaking && speakingMsgId === msgId) {
        stopSpeaking();
      } else {
        speakText(content, msgId);
      }
    },
    [isSpeaking, speakingMsgId, stopSpeaking, speakText],
  );

  const toggleVoiceResponseMode = useCallback(async () => {
    const nextVal = !voiceResponseEnabled;
    setVoiceResponseEnabled(nextVal);
    if (!nextVal) {
      stopSpeaking();
    }
    try {
      const currentSettings = (await RNFS.exists(SETTINGS_PATH))
        ? JSON.parse(await RNFS.readFile(SETTINGS_PATH, 'utf8'))
        : {};
      await RNFS.writeFile(
        SETTINGS_PATH,
        JSON.stringify({
          ...currentSettings,
          voiceResponseEnabled: nextVal,
        }),
        'utf8',
      );
    } catch {}
  }, [voiceResponseEnabled, stopSpeaking]);

  const switchModel = async (newId: string) => {
    if (newId === selectedModelId && phase === 'ready') return;
    setSettingsOpen(false);
    setSelectedModelId(newId);
    const targetModel = AVAILABLE_MODELS.find(m => m.id === newId);
    let nextText = activeTextModelId;
    let nextVision = activeVisionModelId;
    if (targetModel?.category === 'text') {
      nextText = newId;
      setActiveTextModelId(newId);
    } else if (targetModel?.category === 'vision') {
      nextVision = newId;
      setActiveVisionModelId(newId);
    }
    try {
      await RNFS.writeFile(
        SETTINGS_PATH,
        JSON.stringify({
          systemPrompt,
          selectedModelId: newId,
          voiceResponseEnabled,
          activeTextModelId: nextText,
          activeVisionModelId: nextVision,
        }),
        'utf8',
      );
    } catch {}
    prepareModel(newId);
  };

  const activateModel = async (modelId: string, category: 'text' | 'vision') => {
    if (category === 'text') {
      setActiveTextModelId(modelId);
    } else {
      setActiveVisionModelId(modelId);
    }
    await switchModel(modelId);
  };

  const deleteModel = (model: ModelOption) => {
    const isCurrentlyActive = model.id === selectedModelId && phase === 'ready';
    const otherDownloaded = AVAILABLE_MODELS.find(
      m => m.id !== model.id && downloadedModels[m.id],
    );
    const totalSize = model.size + (model.mmprojSize || 0);

    if (isCurrentlyActive && !otherDownloaded) {
      Alert.alert(
        'Cannot Delete Only Model',
        `This is the only downloaded model currently on your device (${formatSize(totalSize)}).\n\nPlease download another model first before deleting this one so the app can continue working offline.`,
      );
      return;
    }

    Alert.alert(
      'Delete Model Files',
      isCurrentlyActive && otherDownloaded
        ? `This model (${model.name}) is currently active. Deleting it will switch you to ${otherDownloaded.name} and permanently free ${formatSize(totalSize)} of storage.`
        : `Delete ${model.name} from your device? This will permanently free ${formatSize(totalSize)} of storage.`,
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Delete',
          style: 'destructive',
          onPress: async () => {
            try {
              if (isCurrentlyActive && otherDownloaded) {
                await switchModel(otherDownloaded.id);
              }
              const path = `${RNFS.DocumentDirectoryPath}/${model.file}`;
              if (await RNFS.exists(path)) {
                await RNFS.unlink(path);
              }
              const tempPath = `${path}.part`;
              if (await RNFS.exists(tempPath)) {
                await RNFS.unlink(tempPath);
              }
              if (model.mmprojFile) {
                const mmPath = `${RNFS.DocumentDirectoryPath}/${model.mmprojFile}`;
                if (await RNFS.exists(mmPath)) {
                  await RNFS.unlink(mmPath);
                }
                const tempMm = `${mmPath}.part`;
                if (await RNFS.exists(tempMm)) {
                  await RNFS.unlink(tempMm);
                }
              }
              await refreshDownloadedStatus();
              Alert.alert('Model Deleted', `${model.name} was removed from device storage.`);
            } catch (err: any) {
              Alert.alert('Error', err?.message || 'Could not delete model.');
            }
          },
        },
      ],
    );
  };

  const handleRequestDownload = (model: ModelOption) => {
    setDownloadMinimized(false);
    const totalSize = model.size + (model.mmprojSize || 0);
    if (totalSize > 3500000000) {
      Alert.alert(
        `Download ${model.name}?`,
        `This is a high-capability ${formatSize(totalSize)} AI ${model.category === 'vision' ? 'multimodal vision brain' : 'brain'}.\n\n` +
          `• Connect to Wi-Fi\n` +
          `• Requires ~${formatSize(totalSize * 1.2)} free phone storage\n` +
          (model.category === 'vision'
            ? `• Includes both Vision Brain and Vision Projector Eyes for direct screenshot & photo analysis\n\n`
            : `• Runs locally with deep reasoning\n\n`) +
          `Proceed with download?`,
        [
          { text: 'Cancel', style: 'cancel' },
          {
            text: 'Download & Activate',
            onPress: () => {
              switchModel(model.id);
            },
          },
        ],
      );
    } else {
      switchModel(model.id);
    }
  };

  const saveSettings = useCallback(async () => {
    Keyboard.dismiss();
    const nextPrompt = tempPrompt.trim() || SYSTEM_PROMPT;
    setSystemPrompt(nextPrompt);
    setSettingsOpen(false);
    try {
      await RNFS.writeFile(
        SETTINGS_PATH,
        JSON.stringify({
          systemPrompt: nextPrompt,
          selectedModelId,
          voiceResponseEnabled,
          activeTextModelId,
          activeVisionModelId,
        }),
        'utf8',
      );
    } catch {}
  }, [tempPrompt, selectedModelId, voiceResponseEnabled, activeTextModelId, activeVisionModelId]);

  // Voice Recording & Hands-free Conversation (100% Offline Speech Recognition)
  const toggleRecording = async () => {
    if (generating) return;

    if (isRecording) {
      // Stop recording
      if (recordingTimer.current) {
        clearInterval(recordingTimer.current);
        recordingTimer.current = null;
      }
      setIsRecording(false);

      if (silenceTimerRef.current) {
        clearTimeout(silenceTimerRef.current);
        silenceTimerRef.current = null;
      }

      try {
        const { NativeSpeech } = NativeModules;
        if (NativeSpeech && typeof NativeSpeech.stopListening === 'function') {
          const res = await NativeSpeech.stopListening();
          const spoken = cleanSpokenPrompt(res?.text?.trim() || '');
          if (spoken) {
            if (voiceResponseEnabled) {
              // Direct conversation mode: send spoken prompt immediately and speak answer back!
              send(spoken);
            } else {
              setInput(prev => (prev ? `${prev} ${spoken}` : spoken));
            }
          } else {
            Alert.alert(
              'Voice Note',
              'No speech detected. Please speak closer to the microphone.\n\nTip: You can also tap the microphone icon on your keyboard for instant dictation anytime!',
            );
          }
        }
      } catch (err: any) {
        Alert.alert(
          'Voice Notice',
          (err?.message || 'Could not finalize voice recording.') +
            '\n\nTip: You can also tap the microphone key on your keyboard for voice input.',
        );
      }
    } else {
      // Stop any active TTS before listening so mic doesn't capture the speaker
      stopSpeaking();

      // Start recording
      try {
        if (Platform.OS === 'android') {
          const granted = await PermissionsAndroid.request(
            PermissionsAndroid.PERMISSIONS.RECORD_AUDIO,
            {
              title: 'Microphone Permission',
              message:
                'IlyasKaGpt needs access to your microphone for offline voice conversation.',
              buttonPositive: 'Allow',
            },
          );
          if (granted !== PermissionsAndroid.RESULTS.GRANTED) {
            Alert.alert(
              'Microphone Permission',
              'Please allow Microphone access in Android Settings to talk with IlyasKaGpt offline.',
            );
            return;
          }
        }

        const { NativeSpeech } = NativeModules;
        if (NativeSpeech && typeof NativeSpeech.startListening === 'function') {
          if (Platform.OS === 'ios') {
            const perm = await NativeSpeech.requestPermissions();
            if (!perm?.speech || !perm?.microphone) {
              Alert.alert(
                'Microphone Access',
                'Please allow Speech Recognition and Microphone access in iOS Settings to transcribe offline.',
              );
              return;
            }
          }
          await NativeSpeech.startListening('');
          setIsRecording(true);
          setRecordingSeconds(0);
          recordingTimer.current = setInterval(() => {
            setRecordingSeconds(s => s + 1);
          }, 1000);
        } else {
          Alert.alert(
            'Offline Voice Dictation',
            'Offline speech recognition module is ready! Rebuild your app to activate hands-free microphone listening.',
          );
        }
      } catch (err: any) {
        Alert.alert(
          'Voice Notice',
          (err?.message || 'Could not start voice recording.') +
            '\n\nTip: You can also tap the microphone icon on your keyboard for offline dictation anytime!',
        );
      }
    }
  };
  toggleRecordingRef.current = toggleRecording;

  // Helper to extract text from chosen image with feedback
  const processImageAttachment = async (asset: { uri?: string; fileName?: string; fileSize?: number }) => {
    if (!asset.uri) return;
    const tempId = `${Date.now()}`;
    const name = asset.fileName || 'photo.jpg';

    const localUri = await ensureLocalFilePath(asset.uri);
    const isVisionReady = activeModel.category === 'vision' && isMultimodalReadyRef.current;

    // Set initial attachment
    setAttachment({
      id: tempId,
      type: 'image',
      uri: localUri,
      name,
      size: asset.fileSize,
      isExtracting: !isVisionReady,
    });

    // If running with an active Multimodal Vision model, it directly views pixels offline!
    if (isVisionReady) {
      return;
    }

    try {
      const text = await extractTextFromImage(localUri);
      setAttachment(prev => {
        if (!prev || prev.id !== tempId) return prev;
        return {
          ...prev,
          extractedText: text || undefined,
          isExtracting: false,
        };
      });
    } catch {
      setAttachment(prev => {
        if (!prev || prev.id !== tempId) return prev;
        return {
          ...prev,
          isExtracting: false,
        };
      });
    }
  };

  // Attachment handlers
  const handleTakePhoto = async () => {
    if (isPickingMediaRef.current) return;
    isPickingMediaRef.current = true;
    setAttachMenuOpen(false);

    // Wait for attachment sheet Modal dismissal animation on iOS before presenting camera
    if (Platform.OS === 'ios') {
      await new Promise<void>(resolve => setTimeout(() => resolve(), 450));
    }

    try {
      const res = await launchCamera({
        mediaType: 'photo',
        quality: 0.8,
        saveToPhotos: false,
      });
      if (res.didCancel || !res.assets || !res.assets[0]?.uri) return;
      await processImageAttachment(res.assets[0]);
    } catch (e: any) {
      const msg = e?.message || String(e);
      if (!msg.toLowerCase().includes('cancel')) {
        Alert.alert('Camera Notice', msg || 'Could not open camera.');
      }
    } finally {
      isPickingMediaRef.current = false;
    }
  };

  const handleChoosePhoto = async () => {
    if (isPickingMediaRef.current) return;
    isPickingMediaRef.current = true;
    setAttachMenuOpen(false);

    // Wait for attachment sheet Modal dismissal animation on iOS before presenting photo library
    if (Platform.OS === 'ios') {
      await new Promise<void>(resolve => setTimeout(() => resolve(), 450));
    }

    try {
      const res = await launchImageLibrary({
        mediaType: 'photo',
        quality: 0.8,
        selectionLimit: 1,
      });
      if (res.didCancel || !res.assets || !res.assets[0]?.uri) return;
      await processImageAttachment(res.assets[0]);
    } catch (e: any) {
      const msg = e?.message || String(e);
      if (!msg.toLowerCase().includes('cancel')) {
        Alert.alert('Photo Notice', msg || 'Could not select photo.');
      }
    } finally {
      isPickingMediaRef.current = false;
    }
  };

  const handlePickDocument = async () => {
    if (isPickingMediaRef.current) return;
    isPickingMediaRef.current = true;
    setAttachMenuOpen(false);

    // Wait for attachment sheet Modal dismissal animation on iOS before presenting UIDocumentPicker
    // This completely resolves "Warning: previous promise did not settle and you attempted to overwrite it."
    if (Platform.OS === 'ios') {
      await new Promise<void>(resolve => setTimeout(() => resolve(), 450));
    }

    try {
      const [doc] = await pick({
        type: [types.pdf, types.plainText, types.allFiles],
        mode: 'import',
      });
      if (!doc?.uri) return;

      const lower = (doc.name || '').toLowerCase();
      const isImg =
        lower.endsWith('.jpg') ||
        lower.endsWith('.jpeg') ||
        lower.endsWith('.png') ||
        lower.endsWith('.webp') ||
        lower.endsWith('.heic');

      let localUri = doc.uri;
      if (isImg) {
        localUri = await ensureLocalFilePath(doc.uri);
      }

      const isVisionReady = isImg && activeModel.category === 'vision' && isMultimodalReadyRef.current;

      const tempId = `${Date.now()}`;
      setAttachment({
        id: tempId,
        type: isImg ? 'image' : 'document',
        uri: localUri,
        name: doc.name || (isImg ? 'photo.jpg' : 'document.pdf'),
        size: doc.size ?? undefined,
        isExtracting: !isVisionReady,
      });

      let extracted = '';
      try {
        if (isImg) {
          if (!isVisionReady) {
            extracted = await extractTextFromImage(localUri);
          }
        } else if (lower.endsWith('.pdf')) {
          extracted = await extractTextFromPdf(doc.uri);
        } else if (
          lower.endsWith('.txt') ||
          lower.endsWith('.md') ||
          lower.endsWith('.json') ||
          lower.endsWith('.csv') ||
          lower.endsWith('.log') ||
          lower.endsWith('.py') ||
          lower.endsWith('.js') ||
          lower.endsWith('.ts')
        ) {
          let cleanUri = doc.uri;
          if (Platform.OS === 'ios' && cleanUri.startsWith('file://')) {
            cleanUri = cleanUri.replace('file://', '');
          }
          extracted = await RNFS.readFile(cleanUri, 'utf8');
        } else {
          // General document fallback: try reading as utf8 text
          let cleanUri = doc.uri;
          if (Platform.OS === 'ios' && cleanUri.startsWith('file://')) {
            cleanUri = cleanUri.replace('file://', '');
          }
          extracted = await RNFS.readFile(cleanUri, 'utf8');
        }
      } catch (err) {
        console.warn('Text extraction error:', err);
      }

      setAttachment(prev => {
        if (!prev || prev.id !== tempId) return prev;
        return {
          ...prev,
          extractedText: extracted || undefined,
          isExtracting: false,
        };
      });
    } catch (e: any) {
      const msg = e?.message || String(e);
      // Silently ignore user cancel and promise collisions
      if (
        (isErrorWithCode(e) && e.code === errorCodes.OPERATION_CANCELED) ||
        msg.includes('previous promise did not settle') ||
        msg.includes('already in progress') ||
        msg.toLowerCase().includes('cancel')
      ) {
        return;
      }
      Alert.alert('File Notice', msg || 'Could not select document.');
    } finally {
      isPickingMediaRef.current = false;
    }
  };

  // Reliable auto-scroll helper
  const scrollToBottom = useCallback((animated = true) => {
    isUserAtBottomRef.current = true;
    setShowScrollBottomBtn(false);
    setTimeout(() => {
      listRef.current?.scrollToEnd({ animated });
    }, 40);
  }, []);

  // Send message (supports typed text or spoken prompt directly)
  const send = useCallback(
    async (customText?: unknown) => {
      const rawText = cleanSpokenPrompt(
        (typeof customText === 'string' ? customText : input).trim(),
      );
      const currentAttach = attachment;
      const ctx = ctxRef.current;
      if ((!rawText && !currentAttach) || !ctx || generating) {
        return;
      }

      // Stop any active speech before starting new response
      stopSpeaking();

      // Dismiss keyboard immediately to reveal full chat area
      Keyboard.dismiss();

      // Reset user to bottom when sending a new message
      isUserAtBottomRef.current = true;
      setShowScrollBottomBtn(false);

      const text =
        rawText ||
        (currentAttach?.type === 'image'
          ? (currentAttach.extractedText
              ? 'Analyze the text from this image:'
              : 'Analyze this photo:')
          : 'Summarize this document:');
      const userMsg: Message = {
        id: `${Date.now()}u`,
        role: 'user',
        content: text,
        attachment: currentAttach || undefined,
      };
      const botId = `${Date.now()}a`;

      // Sliding context window: keep recent history within safe token budget
      const allHistory = [...messages, userMsg];
      let charBudget = 5000;
      const history: Message[] = [];
      for (let i = allHistory.length - 1; i >= 0; i--) {
        const msg = allHistory[i];
        const len = msg.content.length + (msg.attachment?.extractedText?.length || 0);
        if (charBudget > 0 || i === allHistory.length - 1) {
          history.unshift(msg);
          charBudget -= len;
        }
      }

      setInput('');
      setAttachment(null);
      setSpeed('');
      setGenerating(true);
      const initialList: Message[] = [
        ...messages,
        userMsg,
        { id: botId, role: 'assistant', content: '' },
      ];
      setMessages(initialList);
      persistSessionMessages(initialList);

      // Ensure immediate scroll down to new prompt
      scrollToBottom(true);

      // Build context with attachment for offline AI
      let promptWithAttachment = text;
      if (currentAttach) {
        if (currentAttach.type === 'document' && currentAttach.extractedText) {
          promptWithAttachment = `[Attached Document: "${currentAttach.name}"]\n--- Document Content ---\n${currentAttach.extractedText.slice(0, 3000)}\n--- End Document ---\n\n${text}`;
        } else if (currentAttach.type === 'image') {
          if (currentAttach.extractedText) {
            promptWithAttachment = `[Attached Image: "${currentAttach.name}"]\n--- Extracted Text from Image (OCR) ---\n${currentAttach.extractedText.slice(0, 3000)}\n--- End Extracted Text ---\n\n${text}`;
          } else {
            promptWithAttachment = `[Attached Image: "${currentAttach.name}"]\n(Note: No readable text was detected on this image.)\n\n${text}`;
          }
        } else {
          promptWithAttachment = `[Attached File: "${currentAttach.name}"]\n${text}`;
        }
      }

      // Check if Vision model is active and multimodal is initialized
      const isVisionActive =
        activeModel.category === 'vision' &&
        isMultimodalReadyRef.current &&
        currentAttach?.type === 'image' &&
        !!currentAttach.uri;

      let messagesPayload: any[] = [];
      if (isVisionActive && currentAttach) {
        messagesPayload = [
          { role: 'system', content: systemPrompt },
          ...history.slice(0, -1).map(m => ({ role: m.role, content: m.content })),
          {
            role: 'user',
            content: [
              { type: 'text', text: text },
              { type: 'image_url', image_url: { url: currentAttach.uri } },
            ],
          },
        ];
      } else {
        messagesPayload = [
          { role: 'system', content: systemPrompt },
          ...history.map((m, idx) => {
            if (idx === history.length - 1 && currentAttach) {
              return { role: m.role, content: promptWithAttachment };
            }
            return { role: m.role, content: m.content };
          }),
        ];
      }

      let streamed = '';
      let sentenceBuffer = '';
      let sentenceCount = 0;

      // Provide explicit media_paths for llama.rn native compatibility
      const mediaPaths: string[] = [];
      if (isVisionActive && currentAttach?.uri) {
        let clean = currentAttach.uri;
        if (clean.startsWith('file://')) clean = clean.slice(7);
        mediaPaths.push(clean);
      }

      try {
        const result = await ctx.completion(
          {
            messages: messagesPayload,
            ...(mediaPaths.length > 0 ? { media_paths: mediaPaths } : {}),
            n_predict: 512,
            temperature: 0.7,
            top_p: 0.9,
            stop: STOP_WORDS,
          },
          data => {
            streamed += data.token;
            sentenceBuffer += data.token;

            // Zero-latency sentence streaming: speak sentence-by-sentence
            if (voiceResponseEnabled) {
              const match = sentenceBuffer.match(/^(.*?[^0-9\s][.!?])(?:\s+|\n+|$)/);
              if (match && match[1].trim().length >= 8) {
                const completeSentence = match[1].trim();
                sentenceBuffer = sentenceBuffer.slice(match[0].length);
                const cleaned = cleanMarkdownForSpeech(completeSentence);
                if (cleaned && cleaned.length > 3) {
                  speakSentence(cleaned, sentenceCount === 0, botId);
                  sentenceCount++;
                }
              }
            }

            setMessages(prev =>
              prev.map(m => (m.id === botId ? { ...m, content: streamed } : m)),
            );
          },
        );
        const finalText = (result.text || streamed).trim();
        const finalList: Message[] = [
          ...messages,
          userMsg,
          { id: botId, role: 'assistant', content: finalText },
        ];
        setMessages(finalList);
        persistSessionMessages(finalList);
        const tps = result.timings?.predicted_per_second;
        if (tps) {
          setSpeed(`${tps.toFixed(1)} tokens/s`);
        }

        // Only scroll to bottom if user hasn't scrolled up to read earlier text
        if (isUserAtBottomRef.current && !isUserTouchingScrollRef.current) {
          scrollToBottom(true);
        }

        // Speak remaining sentence or fallback full text
        if (voiceResponseEnabled) {
          const remaining = cleanMarkdownForSpeech(sentenceBuffer.trim());
          if (remaining && remaining.length > 2) {
            speakSentence(remaining, sentenceCount === 0, botId);
            sentenceCount++;
          } else if (sentenceCount === 0 && finalText) {
            speakText(finalText, botId);
          }
        }
      } catch (e: any) {
        const errorList: Message[] = [
          ...messages,
          userMsg,
          { id: botId, role: 'assistant', content: `⚠️ ${e?.message ?? String(e)}` },
        ];
        setMessages(errorList);
        persistSessionMessages(errorList);
        if (isUserAtBottomRef.current && !isUserTouchingScrollRef.current) {
          scrollToBottom(true);
        }
      } finally {
        setGenerating(false);
      }
    },
    [
      input,
      attachment,
      messages,
      generating,
      systemPrompt,
      scrollToBottom,
      persistSessionMessages,
      stopSpeaking,
      voiceResponseEnabled,
      speakText,
      speakSentence,
      activeModel.category,
    ],
  );

  const stop = useCallback(() => {
    ctxRef.current?.stopCompletion();
    stopSpeaking();
  }, [stopSpeaking]);

  // Filter sessions by date groupings (matching Image 2)
  const todaySessions = sessions.filter(s => {
    const age = Date.now() - (s.updatedAt || s.createdAt);
    return age < 24 * 60 * 60 * 1000;
  });

  const previous7DaysSessions = sessions.filter(s => {
    const age = Date.now() - (s.updatedAt || s.createdAt);
    return age >= 24 * 60 * 60 * 1000 && age < 7 * 24 * 60 * 60 * 1000;
  });

  const olderSessions = sessions.filter(s => {
    const age = Date.now() - (s.updatedAt || s.createdAt);
    return age >= 7 * 24 * 60 * 60 * 1000;
  });

  // ---- UI -----------------------------------------------------------------
  if ((phase !== 'ready' && !downloadMinimized) || (phase === 'checking' && !ctxRef.current)) {
    return (
      <View style={styles.center}>
        <Text style={styles.logo}>
          <Text style={{ color: C.accent }}>Ilyas</Text>
          <Text style={{ color: '#7C3AED' }}>Ka</Text>
          <Text style={{ color: '#EC4899' }}>Gpt</Text>
        </Text>
        <Text style={styles.subtitle}>{activeDownloadModel.name} · On-Device</Text>

        {phase === 'checking' && <Status text="Checking model…" />}

        {phase === 'downloading' && (
          <View style={styles.card}>
            <View style={styles.downloadCardHeader}>
              <View style={{ flex: 1, marginRight: 8 }}>
                <Text style={styles.cardTitle}>{downloadTitle || `Downloading ${activeDownloadModel.name}`}</Text>
                <Text style={styles.downloadCardSubtitle}>
                  {activeDownloadModel.tag} · Works 100% offline
                </Text>
              </View>
              <Pressable
                style={({ pressed }) => [
                  styles.cardMinimizeIconBtn,
                  pressed && styles.pressed,
                ]}
                onPress={() => setDownloadMinimized(true)}
                hitSlop={8}
                accessibilityLabel="Minimize Download">
                <ChevronDown size={22} color={C.textSecondary} />
              </Pressable>
            </View>

            <View style={styles.barTrack}>
              <View
                style={[
                  styles.barFill,
                  { width: `${Math.max(2, Math.min(100, progress * 100)).toFixed(1)}%` },
                ]}
              />
            </View>

            <View style={styles.progressStatsRow}>
              <Text style={styles.progressPercentText}>
                {(progress * 100).toFixed(1)}%
              </Text>
              <Text style={styles.progressBytesText}>
                {formatDetailedSize(downloaded)} /{' '}
                {formatDetailedSize(
                  downloadTotal ||
                  (downloadTitle.includes('Vision Eyes')
                    ? activeDownloadModel.mmprojSize || activeDownloadModel.size
                    : activeDownloadModel.size),
                )}
              </Text>
            </View>

            {/* Detailed Done & Remaining in MB */}
            {(() => {
              const currentTotal =
                downloadTotal ||
                (downloadTitle.includes('Vision Eyes')
                  ? activeDownloadModel.mmprojSize || activeDownloadModel.size
                  : activeDownloadModel.size);
              const currentRemaining = Math.max(0, currentTotal - downloaded);
              return (
                <View style={styles.mbDetailsRow}>
                  <Text style={styles.mbDetailsText}>
                    Done: <Text style={styles.mbDetailsBold}>{formatMBOnly(downloaded)}</Text>
                  </Text>
                  <Text style={styles.mbDetailsDivider}>•</Text>
                  <Text style={styles.mbDetailsText}>
                    Remaining: <Text style={styles.mbDetailsBold}>{formatMBOnly(currentRemaining)}</Text>
                  </Text>
                </View>
              );
            })()}

            {(downloadSpeed !== '' || downloadEta !== '') && (
              <View style={styles.speedEtaRow}>
                {downloadSpeed ? (
                  <View style={styles.speedBadge}>
                    <Zap size={11} color={C.accent} />
                    <Text style={styles.speedBadgeText}>{downloadSpeed}</Text>
                  </View>
                ) : null}
                {downloadEta ? (
                  <Text style={styles.etaText}>{downloadEta}</Text>
                ) : null}
              </View>
            )}

            <Text style={styles.hint}>
              Keep the app open and Wi-Fi on. Once downloaded, it works 100% offline forever.
            </Text>
            <ActivityIndicator color={C.accent} style={{ marginTop: 10 }} />

            <View style={styles.downloadBtnGroup}>
              <Pressable
                style={({ pressed }) => [
                  styles.minimizeDownloadBtn,
                  pressed && styles.pressed,
                ]}
                onPress={() => setDownloadMinimized(true)}>
                <ChevronDown size={16} color="#FFFFFF" />
                <Text style={styles.minimizeDownloadText}>Minimize to Background</Text>
              </Pressable>

              <Pressable
                style={({ pressed }) => [
                  styles.cancelDownloadBtn,
                  pressed && styles.pressed,
                ]}
                onPress={cancelDownload}>
                <X size={15} color={C.danger} />
                <Text style={styles.cancelDownloadText}>Cancel Download</Text>
              </Pressable>
            </View>
          </View>
        )}

        {phase === 'loading' && <Status text="Loading model into memory…" />}

        {phase === 'error' && (
          <View style={styles.card}>
            <Text style={[styles.cardTitle, { color: C.danger }]}>Download Notice</Text>
            <Text style={styles.cardText}>{error}</Text>
            <View style={{ flexDirection: 'row', gap: 10, marginTop: 14, width: '100%' }}>
              <Pressable
                style={({ pressed }) => [styles.button, { flex: 1 }, pressed && styles.pressed]}
                onPress={() => prepareModel()}>
                <Text style={styles.buttonText}>Retry</Text>
              </Pressable>
              {AVAILABLE_MODELS.some(m => downloadedModels[m.id] && m.id !== activeModel.id) && (
                <Pressable
                  style={({ pressed }) => [
                    styles.button,
                    { flex: 1, backgroundColor: C.surface, borderWidth: 1, borderColor: C.border },
                    pressed && styles.pressed,
                  ]}
                  onPress={() => {
                    const fallback = AVAILABLE_MODELS.find(m => downloadedModels[m.id]);
                    if (fallback) {
                      switchModel(fallback.id);
                    }
                  }}>
                  <Text style={[styles.buttonText, { color: C.text }]}>Use Cached</Text>
                </Pressable>
              )}
            </View>
          </View>
        )}
      </View>
    );
  }

  return (
    <KeyboardAvoidingView
      style={styles.root}
      behavior={Platform.OS === 'ios' ? 'padding' : 'height'}>
      {/* Full-screen Edge-to-Edge Navigation Header */}
      <View style={styles.header}>
        <View style={styles.headerLeft}>
          <Pressable
            style={({ pressed }) => [
              styles.headerIconBtn,
              pressed && styles.pressed,
            ]}
            onPress={() => setDrawerOpen(true)}
            hitSlop={6}
            accessibilityLabel="Open Chat History">
            <Menu size={18} color={C.text} />
          </Pressable>

          <Pressable
            style={({ pressed }) => [
              styles.headerBrandStack,
              pressed && { opacity: 0.75 },
            ]}
            onPress={() => {
              setSettingsTab('models');
              setSettingsOpen(true);
            }}
            hitSlop={4}
            accessibilityLabel="Switch AI Model">
            <View style={styles.headerBrandRow}>
              <View style={styles.headerLogoBadge}>
                <Sparkles size={12} color="#7C3AED" />
              </View>
              <Text style={styles.headerTitle} numberOfLines={1}>
                <Text style={{ color: C.accent }}>Ilyas</Text>
                <Text style={{ color: '#7C3AED' }}>Ka</Text>
                <Text style={{ color: '#EC4899' }}>Gpt</Text>
              </Text>
            </View>

            <View style={styles.headerSubtitleRow}>
              <View
                style={[
                  styles.statusDot,
                  activeModel.category === 'vision' && { backgroundColor: '#7C3AED' },
                ]}
              />
              <Text style={styles.headerSubtitleText} numberOfLines={1}>
                {activeModel.category === 'vision' ? '👁️ ' : '🧠 '}
                {activeModel.id === 'qwen2.5-14b'
                  ? '14B Ultra Genius'
                  : activeModel.id === 'qwen2.5-7b'
                  ? '7B Super Genius'
                  : activeModel.id === 'qwen2.5-3b'
                  ? '3B Smart AI'
                  : activeModel.id === 'qwen2.5-1.5b'
                  ? '1.5B Fast AI'
                  : activeModel.id === 'qwen2-vl-7b'
                  ? '7B Vision Master'
                  : activeModel.id === 'qwen2-vl-2b'
                  ? '2B Vision Fast'
                  : activeModel.name}
              </Text>
              <ChevronDown size={11} color={C.muted} />
            </View>
          </Pressable>
        </View>

        <View style={styles.headerRight}>
          {/* Offline Voice Response Mode Toggle */}
          <Pressable
            style={({ pressed }) => [
              styles.headerIconBtn,
              voiceResponseEnabled && styles.headerVoiceBtnActive,
              pressed && styles.pressed,
            ]}
            onPress={toggleVoiceResponseMode}
            hitSlop={4}
            accessibilityLabel={
              voiceResponseEnabled
                ? 'Voice Response Mode ON (Tap to mute)'
                : 'Voice Response Mode OFF (Tap to enable offline voice answers)'
            }>
            {voiceResponseEnabled ? (
              <Volume2 size={17} color={C.accent} />
            ) : (
              <VolumeX size={17} color={C.muted} />
            )}
          </Pressable>

          <Pressable
            style={({ pressed }) => [
              styles.headerIconBtn,
              pressed && styles.pressed,
            ]}
            onPress={startNewChat}
            hitSlop={4}
            accessibilityLabel="New Chat">
            <Plus size={18} color={C.text} />
          </Pressable>

          <Pressable
            style={({ pressed }) => [
              styles.headerIconBtn,
              pressed && styles.pressed,
            ]}
            onPress={() => {
              setTempPrompt(systemPrompt);
              setSettingsTab('persona');
              setSettingsOpen(true);
            }}
            hitSlop={4}
            accessibilityLabel="Settings">
            <Settings size={17} color={C.text} />
          </Pressable>
        </View>
      </View>

      {/* Mini Download Banner when minimized */}
      {(isDownloadingRef.current || downloadingModelId !== null) && downloadMinimized && (
        <View style={styles.miniDownloadBanner}>
          <View style={styles.miniDownloadLeft}>
            <View style={styles.miniDownloadIconBadge}>
              <Download size={13} color={C.accent} />
            </View>
            <View style={{ flex: 1 }}>
              <View style={styles.miniDownloadTitleRow}>
                <Text style={styles.miniDownloadTitle} numberOfLines={1}>
                  {downloadTitle || `Downloading ${activeDownloadModel.name}`}
                </Text>
                <Text style={styles.miniDownloadPercent}>
                  {(progress * 100).toFixed(0)}%
                </Text>
              </View>
              <Text style={styles.miniDownloadStats} numberOfLines={1}>
                Done: {formatMBOnly(downloaded)} · Left:{' '}
                {formatMBOnly(
                  Math.max(
                    0,
                    (downloadTotal ||
                      (downloadTitle.includes('Vision Eyes')
                        ? activeDownloadModel.mmprojSize || activeDownloadModel.size
                        : activeDownloadModel.size)) - downloaded,
                  ),
                )}
                {downloadSpeed ? ` · ${downloadSpeed}` : ''}
              </Text>
            </View>
          </View>

          <View style={styles.miniDownloadActions}>
            <Pressable
              style={({ pressed }) => [
                styles.miniDownloadBtn,
                pressed && styles.pressed,
              ]}
              onPress={() => setDownloadMinimized(false)}
              hitSlop={6}>
              <Maximize2 size={12} color={C.accent} />
              <Text style={styles.miniDownloadBtnText}>View</Text>
            </Pressable>
            <Pressable
              style={({ pressed }) => [
                styles.miniDownloadSettingsBtn,
                pressed && styles.pressed,
              ]}
              onPress={() => {
                setSettingsTab('downloads');
                setSettingsOpen(true);
              }}
              hitSlop={6}>
              <Settings size={13} color={C.textSecondary} />
            </Pressable>
          </View>
        </View>
      )}

      {/* Chat Messages */}
      <FlatList
        ref={listRef}
        data={messages}
        keyExtractor={m => m.id}
        style={{ flex: 1 }}
        contentContainerStyle={styles.list}
        keyboardDismissMode="on-drag"
        keyboardShouldPersistTaps="handled"
        onScrollBeginDrag={() => {
          isUserTouchingScrollRef.current = true;
        }}
        onScrollEndDrag={() => {
          setTimeout(() => {
            isUserTouchingScrollRef.current = false;
          }, 350);
        }}
        onMomentumScrollEnd={e => {
          isUserTouchingScrollRef.current = false;
          const { layoutMeasurement, contentOffset, contentSize } = e.nativeEvent;
          const isAtBottom =
            layoutMeasurement.height + contentOffset.y >= contentSize.height - 70;
          isUserAtBottomRef.current = isAtBottom;
          setShowScrollBottomBtn(!isAtBottom);
        }}
        onScroll={e => {
          const { layoutMeasurement, contentOffset, contentSize } = e.nativeEvent;
          const isAtBottom =
            layoutMeasurement.height + contentOffset.y >= contentSize.height - 70;
          isUserAtBottomRef.current = isAtBottom;
          setShowScrollBottomBtn(!isAtBottom);
        }}
        scrollEventThrottle={16}
        onContentSizeChange={() => {
          // Only auto-scroll to bottom if the user hasn't scrolled up to read earlier content!
          if (isUserAtBottomRef.current && !isUserTouchingScrollRef.current) {
            scrollToBottom(true);
          }
        }}
        ListEmptyComponent={
          phase !== 'ready' && !ctxRef.current && (isDownloadingRef.current || downloadingModelId !== null) ? (
            <View style={styles.emptyCard}>
              <View style={{ flexDirection: 'row', alignItems: 'center', gap: 10, marginBottom: 6 }}>
                <ActivityIndicator size="small" color={C.accent} />
                <Text style={styles.emptyTitle}>Downloading {activeDownloadModel.name}</Text>
              </View>
              <Text style={styles.emptySubtitle}>
                Model is downloading in the background ({(progress * 100).toFixed(1)}%).
                Once completed, offline AI chat will activate automatically.
              </Text>

              <View style={[styles.barTrack, { marginTop: 12, marginBottom: 8 }]}>
                <View
                  style={[
                    styles.barFill,
                    { width: `${Math.max(2, Math.min(100, progress * 100)).toFixed(1)}%` },
                  ]}
                />
              </View>

              <View style={styles.progressStatsRow}>
                <Text style={styles.progressPercentText}>{(progress * 100).toFixed(1)}%</Text>
                <Text style={styles.progressBytesText}>
                  Done: {formatMBOnly(downloaded)} · Remaining:{' '}
                  {formatMBOnly(
                    Math.max(
                      0,
                      (downloadTotal ||
                        (downloadTitle.includes('Vision Eyes')
                          ? activeDownloadModel.mmprojSize || activeDownloadModel.size
                          : activeDownloadModel.size)) - downloaded,
                    ),
                  )}
                </Text>
              </View>

              <View style={{ flexDirection: 'row', gap: 10, marginTop: 14 }}>
                <Pressable
                  style={({ pressed }) => [styles.button, { flex: 1 }, pressed && styles.pressed]}
                  onPress={() => setDownloadMinimized(false)}>
                  <Text style={styles.buttonText}>Expand Popup</Text>
                </Pressable>
                <Pressable
                  style={({ pressed }) => [
                    styles.button,
                    { flex: 1, backgroundColor: C.surface, borderWidth: 1, borderColor: C.border },
                    pressed && styles.pressed,
                  ]}
                  onPress={() => {
                    setSettingsTab('downloads');
                    setSettingsOpen(true);
                  }}>
                  <Text style={[styles.buttonText, { color: C.text }]}>Open Downloads</Text>
                </Pressable>
              </View>
            </View>
          ) : (
            <View style={styles.emptyCard}>
              <Text style={styles.emptyTitle}>What can I help you build today?</Text>
              <Text style={styles.emptySubtitle}>
                Select an action suggestion or type anything below:
              </Text>

              <View style={styles.chipsGrid}>
                <Pressable
                  style={({ pressed }) => [styles.suggestionChip, pressed && styles.pressed]}
                  onPress={() => setInput('Explain this code:\n')}>
                  <Code2 size={15} color={C.accent} />
                  <Text style={styles.chipText}>Explain Code</Text>
                </Pressable>

                <Pressable
                  style={({ pressed }) => [styles.suggestionChip, pressed && styles.pressed]}
                  onPress={() => setInput('Summarize the key points of:\n')}>
                  <FileText size={15} color={C.accent} />
                  <Text style={styles.chipText}>Summarize</Text>
                </Pressable>

                <Pressable
                  style={({ pressed }) => [styles.suggestionChip, pressed && styles.pressed]}
                  onPress={() => setInput('Draft a professional email about:\n')}>
                  <Mail size={15} color={C.accent} />
                  <Text style={styles.chipText}>Write Email</Text>
                </Pressable>

                <Pressable
                  style={({ pressed }) => [styles.suggestionChip, pressed && styles.pressed]}
                  onPress={() => setInput('Brainstorm creative ideas for:\n')}>
                  <Lightbulb size={15} color={C.accent} />
                  <Text style={styles.chipText}>Brainstorm</Text>
                </Pressable>
              </View>
            </View>
          )
        }
        renderItem={({ item }) => {
          const isUser = item.role === 'user';
          const isUserMedia = isUser && !!item.attachment;

          return (
            <View
              style={[
                styles.bubble,
                isUser
                  ? isUserMedia
                    ? styles.userMediaBubble
                    : styles.userBubble
                  : styles.botBubble,
              ]}>
              {item.role === 'assistant' && (
                <View style={styles.botHeader}>
                  <View style={styles.botHeaderLeft}>
                    <View style={styles.botBadge}>
                      <Sparkles size={10} color={C.accent} />
                    </View>
                    <Text style={styles.botName}>
                      <Text style={{ color: C.accent }}>Ilyas</Text>
                      <Text style={{ color: '#7C3AED' }}>KaGpt</Text>
                    </Text>
                  </View>
                  {!!item.content && (
                    <Pressable
                      style={({ pressed }) => [
                        styles.botSpeakBtn,
                        isSpeaking && speakingMsgId === item.id && styles.botSpeakBtnActive,
                        pressed && styles.pressed,
                      ]}
                      onPress={() => speakMessage(item.id, item.content)}
                      hitSlop={8}
                      accessibilityLabel={
                        isSpeaking && speakingMsgId === item.id
                          ? 'Stop reading'
                          : 'Read answer aloud offline'
                      }>
                      {isSpeaking && speakingMsgId === item.id ? (
                        <VolumeX size={13} color={C.danger} />
                      ) : (
                        <Volume2 size={13} color={C.textSecondary} />
                      )}
                    </Pressable>
                  )}
                </View>
              )}

              {/* Attached media inside bubble (Clean white container, no blue background) */}
              {item.attachment && (
                <View style={styles.bubbleAttachmentBox}>
                  {item.attachment.type === 'image' ? (
                    <RNImage
                      source={{ uri: item.attachment.uri }}
                      style={styles.bubbleImage}
                      resizeMode="cover"
                    />
                  ) : (
                    <View
                      style={[
                        styles.bubbleDocCard,
                        isUser && !isUserMedia && styles.bubbleDocCardUser,
                      ]}>
                      <View
                        style={[
                          styles.bubbleDocIcon,
                          isUser && !isUserMedia && styles.bubbleDocIconUser,
                        ]}>
                        <FileText
                          size={18}
                          color={isUser && !isUserMedia ? '#FFFFFF' : C.accent}
                        />
                      </View>
                      <View style={{ flex: 1 }}>
                        <Text
                          style={[
                            styles.bubbleDocName,
                            isUser && !isUserMedia && { color: '#FFFFFF' },
                          ]}
                          numberOfLines={1}>
                          {item.attachment.name}
                        </Text>
                        <Text
                          style={[
                            styles.bubbleDocMeta,
                            isUser && !isUserMedia && {
                              color: 'rgba(255,255,255,0.7)',
                            },
                          ]}>
                          PDF Document ·{' '}
                          {item.attachment.size
                            ? formatMB(item.attachment.size)
                            : 'File'}
                        </Text>
                      </View>
                    </View>
                  )}
                </View>
              )}

              <MessageContent
                content={item.content}
                role={item.role}
                isMediaUser={isUserMedia}
              />
            </View>
          );
        }}
      />

      {!!speed && <Text style={styles.speed}>{speed}</Text>}

      {/* Floating Scroll to Bottom Indicator */}
      {showScrollBottomBtn && (
        <Pressable
          style={({ pressed }) => [
            styles.floatingScrollBtn,
            pressed && styles.pressed,
          ]}
          onPress={() => scrollToBottom(true)}
          accessibilityLabel="Scroll to latest response">
          <ChevronDown size={15} color={C.accent} />
          {generating && <Text style={styles.floatingScrollText}>Generating…</Text>}
        </Pressable>
      )}

      {/* Floating Capsule Input Bar */}
      <View style={styles.inputContainer}>
        {/* Offline Voice Recording Status Banner */}
        {isRecording && (
          <View style={styles.recordingBar}>
            <View style={styles.recordingPulse} />
            <Text style={styles.recordingText}>
              Listening offline ({Math.floor(recordingSeconds / 60)}:
              {(recordingSeconds % 60).toString().padStart(2, '0')})… Tap mic to finish
            </Text>
          </View>
        )}

        {/* Attachment preview if an image/doc is picked */}
        {attachment && (
          <View style={styles.previewBar}>
            {attachment.type === 'image' ? (
              <RNImage
                source={{ uri: attachment.uri }}
                style={styles.previewThumb}
              />
            ) : (
              <View style={styles.previewDocIcon}>
                <FileText size={16} color={C.accent} />
              </View>
            )}
            <View style={styles.previewInfo}>
              <Text style={styles.previewName} numberOfLines={1}>
                {attachment.name}
              </Text>
              <Text style={styles.previewSub} numberOfLines={1}>
                {attachment.isExtracting
                  ? 'Analyzing attachment…'
                  : attachment.type === 'image'
                  ? activeModel.category === 'vision' && isMultimodalReady
                    ? '👁️ Multimodal Vision ready (Direct pixel analysis)'
                    : attachment.extractedText
                    ? `Text extracted: "${attachment.extractedText.slice(0, 25)}…"`
                    : 'Photo attached · Text AI mode (no OCR text found)'
                  : 'Document attached' + (attachment.extractedText ? ' · Text extracted' : '')}
              </Text>
            </View>
            {attachment.type === 'image' && activeModel.category !== 'vision' && (
              (() => {
                const downloadedVision = AVAILABLE_MODELS.find(
                  m => m.id === activeVisionModelId && downloadedModels[m.id],
                ) || AVAILABLE_MODELS.find(
                  m => m.category === 'vision' && downloadedModels[m.id],
                );
                return downloadedVision ? (
                  <Pressable
                    style={styles.quickVisionBtn}
                    onPress={() => switchModel(downloadedVision.id)}
                    hitSlop={4}>
                    <Camera size={11} color="#FFFFFF" />
                    <Text style={styles.quickVisionBtnText}>
                      Switch to {downloadedVision.name.replace('Vision Master', 'Vision').replace('Vision Fast', 'Vision')}
                    </Text>
                  </Pressable>
                ) : (
                  <Pressable
                    style={styles.quickVisionBtnOutline}
                    onPress={() => {
                      setModelCategoryFilter('vision');
                      setSettingsTab('models');
                      setSettingsOpen(true);
                    }}
                    hitSlop={4}>
                    <Camera size={11} color={C.accent} />
                    <Text style={styles.quickVisionBtnOutlineText}>Get Vision AI</Text>
                  </Pressable>
                );
              })()
            )}
            {attachment.isExtracting && (
              <ActivityIndicator size="small" color={C.accent} style={{ marginRight: 6 }} />
            )}
            <Pressable
              style={styles.previewCloseBtn}
              onPress={() => setAttachment(null)}
              hitSlop={8}>
              <X size={15} color={C.muted} />
            </Pressable>
          </View>
        )}

        <View style={styles.inputCard}>
          {/* Plus / Attach Button */}
          <Pressable
            style={({ pressed }) => [
              styles.attachBtn,
              pressed && styles.pressed,
            ]}
            onPress={() => setAttachMenuOpen(true)}
            accessibilityLabel="Attach photo or document">
            <Plus size={20} color={C.textSecondary} />
          </Pressable>

          <TextInput
            style={styles.input}
            value={input}
            onChangeText={setInput}
            placeholder={
              attachment
                ? 'Add instructions or tap Send…'
                : 'Ask IlyasKaGpt…'
            }
            placeholderTextColor={C.muted}
            multiline
            editable={!generating}
          />

          {/* Offline Mic / Dictation Button */}
          <Pressable
            style={({ pressed }) => [
              styles.micBtn,
              isRecording && styles.micBtnActive,
              pressed && styles.pressed,
            ]}
            onPress={toggleRecording}
            accessibilityLabel={isRecording ? 'Stop voice recording' : 'Start offline voice recording'}>
            {isRecording ? (
              <MicOff size={17} color="#FFFFFF" />
            ) : (
              <Mic size={17} color={C.textSecondary} />
            )}
          </Pressable>

          {generating ? (
            <Pressable
              style={({ pressed }) => [
                styles.sendBtn,
                styles.stopBtn,
                pressed && styles.pressed,
              ]}
              onPress={stop}>
              <Square size={14} color="#FFFFFF" />
            </Pressable>
          ) : (
            <Pressable
              style={({ pressed }) => [
                styles.sendBtn,
                !input.trim() && !attachment && styles.disabled,
                pressed && styles.pressed,
              ]}
              onPress={() => send()}
              disabled={!input.trim() && !attachment}>
              <Send size={15} color="#FFFFFF" style={{ marginLeft: 2 }} />
            </Pressable>
          )}
        </View>
      </View>

      {/* Attachment Bottom Sheet Modal */}
      <Modal
        visible={attachMenuOpen}
        animationType="slide"
        transparent
        onRequestClose={() => setAttachMenuOpen(false)}>
        <Pressable
          style={styles.attachOverlay}
          onPress={() => setAttachMenuOpen(false)}>
          <View style={styles.attachSheet}>
            <View style={styles.attachHeader}>
              <Text style={styles.attachTitle}>Attach to IlyasKaGpt</Text>
              <Pressable
                onPress={() => setAttachMenuOpen(false)}
                hitSlop={8}>
                <X size={18} color={C.muted} />
              </Pressable>
            </View>

            <View style={styles.attachOptionsRow}>
              <Pressable
                style={({ pressed }) => [
                  styles.attachOptionCard,
                  pressed && styles.pressed,
                ]}
                onPress={handleTakePhoto}>
                <View style={[styles.attachIconCircle, { backgroundColor: '#EFF6FF' }]}>
                  <Camera size={22} color="#2563EB" />
                </View>
                <Text style={styles.attachOptionTitle}>Take Photo</Text>
                <Text style={styles.attachOptionDesc}>Use Camera</Text>
              </Pressable>

              <Pressable
                style={({ pressed }) => [
                  styles.attachOptionCard,
                  pressed && styles.pressed,
                ]}
                onPress={handleChoosePhoto}>
                <View style={[styles.attachIconCircle, { backgroundColor: '#F5F3FF' }]}>
                  <ImageIcon size={22} color="#7C3AED" />
                </View>
                <Text style={styles.attachOptionTitle}>Photo Library</Text>
                <Text style={styles.attachOptionDesc}>Select Image</Text>
              </Pressable>

              <Pressable
                style={({ pressed }) => [
                  styles.attachOptionCard,
                  pressed && styles.pressed,
                ]}
                onPress={handlePickDocument}>
                <View style={[styles.attachIconCircle, { backgroundColor: '#ECFDF5' }]}>
                  <FileUp size={22} color="#059669" />
                </View>
                <Text style={styles.attachOptionTitle}>Upload PDF</Text>
                <Text style={styles.attachOptionDesc}>From Files</Text>
              </Pressable>
            </View>
          </View>
        </Pressable>
      </Modal>

      {/* Settings Modal (Light Theme) */}
      <Modal
        visible={settingsOpen}
        animationType="slide"
        transparent
        onRequestClose={() => {
          Keyboard.dismiss();
          setSettingsOpen(false);
        }}>
        <KeyboardAvoidingView
          behavior={Platform.OS === 'ios' ? 'padding' : undefined}
          style={styles.modalOverlay}>
          {/* Dimmed backdrop - tap to dismiss keyboard or close modal */}
          <Pressable
            style={styles.modalBackdrop}
            onPress={() => {
              Keyboard.dismiss();
              if (!keyboardVisible) {
                setSettingsOpen(false);
              }
            }}
          />

          <View
            style={[
              styles.modalContent,
              keyboardVisible && styles.modalContentKeyboard,
            ]}>
            <View style={styles.modalHeader}>
              <View style={styles.modalTitleRow}>
                {settingsTab === 'models' ? (
                  <Cpu size={18} color={C.accent} />
                ) : settingsTab === 'downloads' ? (
                  <HardDrive size={18} color={C.accent} />
                ) : (
                  <Settings size={18} color={C.accent} />
                )}
                <Text style={styles.modalTitle}>
                  {settingsTab === 'models'
                    ? 'On-Device AI Models'
                    : settingsTab === 'downloads'
                    ? 'Downloads & Active Brains'
                    : 'Personalization & Persona'}
                </Text>
              </View>
              <View style={styles.modalHeaderActions}>
                {/* Small button to close/hide keyboard */}
                {keyboardVisible && (
                  <Pressable
                    style={styles.dismissKeyboardBtn}
                    onPress={() => Keyboard.dismiss()}
                    hitSlop={8}
                    accessibilityLabel="Close Keyboard">
                    <ChevronDown size={14} color={C.accent} />
                    <Text style={styles.dismissKeyboardText}>Hide Keyboard</Text>
                  </Pressable>
                )}
                <Pressable
                  style={styles.modalCloseBtn}
                  onPress={() => {
                    Keyboard.dismiss();
                    setSettingsOpen(false);
                  }}
                  hitSlop={8}
                  accessibilityLabel="Close Modal">
                  <X size={20} color={C.muted} />
                </Pressable>
              </View>
            </View>

            {/* Segmented Tab Bar - 3 Tabs */}
            <View style={styles.tabContainer}>
              <Pressable
                style={[
                  styles.tabButton,
                  settingsTab === 'models' && styles.tabButtonActive,
                ]}
                onPress={() => {
                  Keyboard.dismiss();
                  setSettingsTab('models');
                }}>
                <Cpu
                  size={14}
                  color={settingsTab === 'models' ? C.accent : C.textSecondary}
                />
                <Text
                  style={[
                    styles.tabButtonText,
                    settingsTab === 'models' && styles.tabButtonTextActive,
                  ]}>
                  Model Hub
                </Text>
              </Pressable>

              <Pressable
                style={[
                  styles.tabButton,
                  settingsTab === 'downloads' && styles.tabButtonActive,
                ]}
                onPress={() => {
                  Keyboard.dismiss();
                  setSettingsTab('downloads');
                  refreshDownloadedStatus();
                }}>
                <HardDrive
                  size={14}
                  color={settingsTab === 'downloads' ? C.accent : C.textSecondary}
                />
                <Text
                  style={[
                    styles.tabButtonText,
                    settingsTab === 'downloads' && styles.tabButtonTextActive,
                  ]}>
                  Downloads
                </Text>
                {(isDownloadingRef.current || downloadingModelId !== null) && (
                  <View style={styles.tabActiveDot} />
                )}
              </Pressable>

              <Pressable
                style={[
                  styles.tabButton,
                  settingsTab === 'persona' && styles.tabButtonActive,
                ]}
                onPress={() => {
                  Keyboard.dismiss();
                  setSettingsTab('persona');
                }}>
                <Settings
                  size={14}
                  color={settingsTab === 'persona' ? C.accent : C.textSecondary}
                />
                <Text
                  style={[
                    styles.tabButtonText,
                    settingsTab === 'persona' && styles.tabButtonTextActive,
                  ]}>
                  Persona
                </Text>
              </Pressable>
            </View>

            <ScrollView
              showsVerticalScrollIndicator={false}
              keyboardDismissMode="on-drag"
              keyboardShouldPersistTaps="handled">
              {settingsTab === 'models' ? (
                /* Tab 1: On-Device AI Model Hub with Text and Vision Categories */
                <View>
                  <View style={styles.modelHubHeader}>
                    <Text style={styles.modelSectionTitle}>Available Offline Brains</Text>
                    <Text style={styles.settingHint}>
                      Select or download models by category. Currently running {activeModel.name}.
                    </Text>
                  </View>

                  {/* Category Switcher Tabs */}
                  <View style={styles.categoryTabRow}>
                    <Pressable
                      style={[
                        styles.categoryTabBtn,
                        modelCategoryFilter === 'text' && styles.categoryTabBtnActive,
                      ]}
                      onPress={() => setModelCategoryFilter('text')}>
                      <Brain
                        size={14}
                        color={modelCategoryFilter === 'text' ? '#FFFFFF' : C.accent}
                      />
                      <Text
                        style={[
                          styles.categoryTabBtnText,
                          modelCategoryFilter === 'text' && styles.categoryTabBtnTextActive,
                        ]}>
                        Text Brains ({AVAILABLE_MODELS.filter(m => m.category === 'text').length})
                      </Text>
                    </Pressable>

                    <Pressable
                      style={[
                        styles.categoryTabBtn,
                        modelCategoryFilter === 'vision' && styles.categoryTabBtnActive,
                      ]}
                      onPress={() => setModelCategoryFilter('vision')}>
                      <Camera
                        size={14}
                        color={modelCategoryFilter === 'vision' ? '#FFFFFF' : '#7C3AED'}
                      />
                      <Text
                        style={[
                          styles.categoryTabBtnText,
                          modelCategoryFilter === 'vision' && styles.categoryTabBtnTextActive,
                        ]}>
                        Vision Brains ({AVAILABLE_MODELS.filter(m => m.category === 'vision').length})
                      </Text>
                    </Pressable>
                  </View>

                  <View style={{ gap: 10, marginBottom: 16 }}>
                    {AVAILABLE_MODELS.filter(m => m.category === modelCategoryFilter).map(m => {
                      const isSelected = selectedModelId === m.id;
                      const isDownloaded = downloadedModels[m.id] === true;
                      const isActive = isSelected && phase === 'ready';
                      const isCurrentlyDownloading =
                        (isDownloadingRef.current || downloadingModelId !== null) &&
                        (m.id === downloadingModelId || m.id === activeDownloadModelIdRef.current);

                      return (
                        <View
                          key={m.id}
                          style={[
                            styles.modelCard,
                            isActive && styles.modelCardActive,
                          ]}>
                          <View style={styles.modelCardTop}>
                            <View style={{ flex: 1, marginRight: 8 }}>
                              <Text
                                style={[
                                  styles.modelCardName,
                                  isActive && { color: C.accent },
                                ]}>
                                {m.name}
                              </Text>
                            </View>
                            <View
                              style={[
                                styles.modelTag,
                                isActive && styles.modelTagActive,
                              ]}>
                              <Text
                                style={[
                                  styles.modelTagText,
                                  isActive && { color: '#FFFFFF' },
                                ]}>
                                {m.tag}
                              </Text>
                            </View>
                          </View>

                          <Text style={styles.modelCardDesc}>{m.description}</Text>

                          {/* Status badge & action buttons */}
                          <View style={styles.modelActionRow}>
                            {isCurrentlyDownloading ? (
                              <View style={styles.modelDownloadingBadge}>
                                <ActivityIndicator size="small" color={C.accent} style={{ transform: [{ scale: 0.7 }] }} />
                                <Text style={styles.modelDownloadingBadgeText}>
                                  Downloading {(progress * 100).toFixed(0)}%
                                </Text>
                              </View>
                            ) : isActive ? (
                              <View style={styles.modelActiveBadge}>
                                <Check size={12} color="#059669" />
                                <Text style={styles.modelActiveBadgeText}>Active & Running</Text>
                              </View>
                            ) : isDownloaded ? (
                              <View style={styles.modelDownloadedBadge}>
                                <HardDrive size={12} color={C.accent} />
                                <Text style={styles.modelDownloadedBadgeText}>Downloaded</Text>
                              </View>
                            ) : (
                              <View style={styles.modelNotDownloadedBadge}>
                                <Text style={styles.modelNotDownloadedText}>Not on device</Text>
                              </View>
                            )}

                            <View style={styles.modelBtnGroup}>
                              {isCurrentlyDownloading ? (
                                <Pressable
                                  style={({ pressed }) => [
                                    styles.modelSwitchBtn,
                                    pressed && styles.pressed,
                                  ]}
                                  onPress={() => setSettingsTab('downloads')}>
                                  <HardDrive size={12} color="#FFFFFF" />
                                  <Text style={styles.modelSwitchBtnText}>View Progress</Text>
                                </Pressable>
                              ) : isDownloaded ? (
                                <>
                                  {!isActive && (
                                    <Pressable
                                      style={({ pressed }) => [
                                        styles.modelSwitchBtn,
                                        pressed && styles.pressed,
                                      ]}
                                      onPress={() => switchModel(m.id)}>
                                      <Zap size={12} color="#FFFFFF" />
                                      <Text style={styles.modelSwitchBtnText}>Switch</Text>
                                    </Pressable>
                                  )}
                                  <Pressable
                                    style={({ pressed }) => [
                                      styles.modelDeleteBtn,
                                      pressed && styles.pressed,
                                    ]}
                                    onPress={() => deleteModel(m)}
                                    hitSlop={6}
                                    accessibilityLabel={`Delete ${m.name}`}>
                                    <Trash2 size={13} color={C.danger} />
                                  </Pressable>
                                </>
                              ) : (
                                <Pressable
                                  style={({ pressed }) => [
                                    styles.modelDownloadBtn,
                                    pressed && styles.pressed,
                                  ]}
                                  onPress={() => handleRequestDownload(m)}>
                                  <Download size={13} color={C.accent} />
                                  <Text style={styles.modelDownloadBtnText}>
                                    Download & Use
                                  </Text>
                                </Pressable>
                              )}
                            </View>
                          </View>
                        </View>
                      );
                    })}
                  </View>
                </View>
              ) : settingsTab === 'downloads' ? (
                /* Tab 2: Downloads & Active Models */
                <View style={{ gap: 16, paddingBottom: 24 }}>
                  {/* Current Download Progress Card */}
                  <View style={styles.downloadsSectionCard}>
                    <View style={styles.downloadsSectionHeaderRow}>
                      <View style={{ flexDirection: 'row', alignItems: 'center', gap: 6 }}>
                        <Download size={16} color={C.accent} />
                        <Text style={styles.downloadsSectionTitle}>Current Download Progress</Text>
                      </View>
                      {(isDownloadingRef.current || downloadingModelId !== null) && (
                        <View style={styles.downloadingLiveBadge}>
                          <ActivityIndicator size="small" color={C.accent} style={{ transform: [{ scale: 0.65 }] }} />
                          <Text style={styles.downloadingLiveBadgeText}>Live</Text>
                        </View>
                      )}
                    </View>

                    {(isDownloadingRef.current || downloadingModelId !== null) ? (
                      <View style={{ gap: 10, marginTop: 4 }}>
                        <View style={{ flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' }}>
                          <View style={{ flex: 1, marginRight: 8 }}>
                            <Text style={styles.dlModelName}>
                              {activeDownloadModel.name}
                            </Text>
                            <Text style={styles.dlPartTitle}>
                              {downloadTitle || (activeDownloadModel.category === 'vision' ? 'Downloading Model Weights (Brain 1/2)' : 'Downloading Model Weights')}
                            </Text>
                          </View>
                          <View style={[styles.modelTag, { backgroundColor: activeDownloadModel.category === 'vision' ? '#F3E8FF' : C.accentSoft }]}>
                            <Text style={[styles.modelTagText, { color: activeDownloadModel.category === 'vision' ? '#7C3AED' : C.accent }]}>
                              {activeDownloadModel.category === 'vision' ? 'Vision AI' : 'Text AI'}
                            </Text>
                          </View>
                        </View>

                        {/* Progress Bar */}
                        <View style={styles.barTrack}>
                          <View
                            style={[
                              styles.barFill,
                              { width: `${Math.max(2, Math.min(100, progress * 100)).toFixed(1)}%` },
                            ]}
                          />
                        </View>

                        <View style={{ flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' }}>
                          <Text style={styles.progressPercentText}>
                            {(progress * 100).toFixed(1)}%
                          </Text>
                          {(downloadSpeed || downloadEta) && (
                            <View style={styles.speedEtaRow}>
                              {downloadSpeed ? (
                                <View style={styles.speedBadge}>
                                  <Zap size={11} color={C.accent} />
                                  <Text style={styles.speedBadgeText}>{downloadSpeed}</Text>
                                </View>
                              ) : null}
                              {downloadEta ? (
                                <Text style={styles.etaText}>{downloadEta}</Text>
                              ) : null}
                            </View>
                          )}
                        </View>

                        {/* MB Done & MB Remaining Stat Cards */}
                        {(() => {
                          const currentTotal =
                            downloadTotal ||
                            (downloadTitle.includes('Vision Eyes')
                              ? activeDownloadModel.mmprojSize || activeDownloadModel.size
                              : activeDownloadModel.size);
                          const currentRemaining = Math.max(0, currentTotal - downloaded);
                          return (
                            <View style={styles.statsCardGrid}>
                              <View style={styles.statBox}>
                                <Text style={styles.statBoxLabel}>MB Done</Text>
                                <Text style={styles.statBoxValue}>{formatMBOnly(downloaded)}</Text>
                                <Text style={styles.statBoxSub}>({formatDetailedSize(downloaded)})</Text>
                              </View>

                              <View style={styles.statBox}>
                                <Text style={styles.statBoxLabel}>MB Remaining</Text>
                                <Text style={[styles.statBoxValue, { color: '#D97706' }]}>
                                  {formatMBOnly(currentRemaining)}
                                </Text>
                                <Text style={styles.statBoxSub}>({formatDetailedSize(currentRemaining)})</Text>
                              </View>

                              <View style={styles.statBox}>
                                <Text style={styles.statBoxLabel}>Total</Text>
                                <Text style={styles.statBoxValue}>{formatMBOnly(currentTotal)}</Text>
                                <Text style={styles.statBoxSub}>({formatDetailedSize(currentTotal)})</Text>
                              </View>
                            </View>
                          );
                        })()}

                        {/* Actions */}
                        <View style={{ flexDirection: 'row', gap: 10, marginTop: 4 }}>
                          <Pressable
                            style={({ pressed }) => [
                              styles.downloadsActionBtn,
                              { flex: 1, backgroundColor: C.accent },
                              pressed && styles.pressed,
                            ]}
                            onPress={() => {
                              setSettingsOpen(false);
                              setDownloadMinimized(false);
                            }}>
                            <Maximize2 size={13} color="#FFFFFF" />
                            <Text style={[styles.downloadsActionBtnText, { color: '#FFFFFF' }]}>
                              Maximize Popup
                            </Text>
                          </Pressable>

                          <Pressable
                            style={({ pressed }) => [
                              styles.downloadsActionBtn,
                              { flex: 1, backgroundColor: '#FEF2F2', borderWidth: 1, borderColor: '#FEE2E2' },
                              pressed && styles.pressed,
                            ]}
                            onPress={cancelDownload}>
                            <X size={13} color={C.danger} />
                            <Text style={[styles.downloadsActionBtnText, { color: C.danger }]}>
                              Cancel Download
                            </Text>
                          </Pressable>
                        </View>
                      </View>
                    ) : (
                      <View style={styles.noDownloadContainer}>
                        <Check size={20} color="#059669" />
                        <View style={{ flex: 1 }}>
                          <Text style={styles.noDownloadTitle}>No active downloads</Text>
                          <Text style={styles.noDownloadDesc}>
                            All downloaded models are stored locally on your device for 100% offline use.
                          </Text>
                        </View>
                      </View>
                    )}
                  </View>

                  {/* 2. Downloaded Text Models (Active selection) */}
                  <View style={styles.downloadsSectionCard}>
                    <View style={styles.downloadsSectionHeaderRow}>
                      <View style={{ flexDirection: 'row', alignItems: 'center', gap: 6 }}>
                        <Brain size={16} color={C.accent} />
                        <Text style={styles.downloadsSectionTitle}>Downloaded Text Models</Text>
                      </View>
                      <View style={styles.countBadge}>
                        <Text style={styles.countBadgeText}>
                          {AVAILABLE_MODELS.filter(m => m.category === 'text' && downloadedModels[m.id]).length} downloaded
                        </Text>
                      </View>
                    </View>
                    <Text style={styles.downloadsSectionSubtitle}>
                      Select which text model is currently active for offline conversation & reasoning.
                    </Text>

                    {AVAILABLE_MODELS.filter(m => m.category === 'text' && downloadedModels[m.id]).length === 0 ? (
                      <View style={styles.emptyModelCategoryCard}>
                        <Text style={styles.emptyModelCategoryText}>
                          No text models downloaded yet.
                        </Text>
                        <Pressable
                          style={styles.browseModelsChip}
                          onPress={() => {
                            setModelCategoryFilter('text');
                            setSettingsTab('models');
                          }}>
                          <Plus size={12} color={C.accent} />
                          <Text style={styles.browseModelsChipText}>Download Text Model</Text>
                        </Pressable>
                      </View>
                    ) : (
                      <View style={{ gap: 8, marginTop: 4 }}>
                        {AVAILABLE_MODELS.filter(m => m.category === 'text' && downloadedModels[m.id]).map(m => {
                          const isCurrentlyRunning = m.id === selectedModelId && phase === 'ready';
                          const isSelectedText = m.id === activeTextModelId;

                          return (
                            <View
                              key={m.id}
                              style={[
                                styles.downloadedItemCard,
                                isCurrentlyRunning && styles.downloadedItemCardActive,
                              ]}>
                              <View style={styles.downloadedItemTop}>
                                <View style={{ flex: 1, marginRight: 8 }}>
                                  <Text style={[styles.downloadedItemName, isCurrentlyRunning && { color: C.accent }]}>
                                    {m.name}
                                  </Text>
                                  <Text style={styles.downloadedItemDesc} numberOfLines={2}>
                                    {m.description}
                                  </Text>
                                </View>
                                <View style={styles.modelTag}>
                                  <Text style={styles.modelTagText}>{formatSize(m.size)}</Text>
                                </View>
                              </View>

                              <View style={styles.downloadedItemBottom}>
                                {isCurrentlyRunning ? (
                                  <View style={styles.modelActiveBadge}>
                                    <Check size={12} color="#059669" />
                                    <Text style={styles.modelActiveBadgeText}>Active in Chat</Text>
                                  </View>
                                ) : isSelectedText ? (
                                  <View style={styles.modelSelectedBadge}>
                                    <Check size={12} color={C.accent} />
                                    <Text style={styles.modelSelectedBadgeText}>Selected for Text</Text>
                                  </View>
                                ) : (
                                  <View style={styles.modelCachedBadge}>
                                    <HardDrive size={11} color={C.textSecondary} />
                                    <Text style={styles.modelCachedBadgeText}>Ready Offline</Text>
                                  </View>
                                )}

                                <View style={{ flexDirection: 'row', alignItems: 'center', gap: 6 }}>
                                  {!isCurrentlyRunning && (
                                    <Pressable
                                      style={({ pressed }) => [
                                        styles.activateBtn,
                                        pressed && styles.pressed,
                                      ]}
                                      onPress={() => activateModel(m.id, 'text')}>
                                      <Zap size={11} color="#FFFFFF" />
                                      <Text style={styles.activateBtnText}>Activate</Text>
                                    </Pressable>
                                  )}
                                  <Pressable
                                    style={({ pressed }) => [
                                      styles.modelDeleteBtn,
                                      pressed && styles.pressed,
                                    ]}
                                    onPress={() => deleteModel(m)}
                                    hitSlop={6}
                                    accessibilityLabel={`Delete ${m.name}`}>
                                    <Trash2 size={13} color={C.danger} />
                                  </Pressable>
                                </View>
                              </View>
                            </View>
                          );
                        })}
                      </View>
                    )}
                  </View>

                  {/* 3. Downloaded Vision Models (Active selection) */}
                  <View style={styles.downloadsSectionCard}>
                    <View style={styles.downloadsSectionHeaderRow}>
                      <View style={{ flexDirection: 'row', alignItems: 'center', gap: 6 }}>
                        <Camera size={16} color="#7C3AED" />
                        <Text style={styles.downloadsSectionTitle}>Downloaded Vision Models</Text>
                      </View>
                      <View style={[styles.countBadge, { backgroundColor: '#F3E8FF' }]}>
                        <Text style={[styles.countBadgeText, { color: '#7C3AED' }]}>
                          {AVAILABLE_MODELS.filter(m => m.category === 'vision' && downloadedModels[m.id]).length} downloaded
                        </Text>
                      </View>
                    </View>
                    <Text style={styles.downloadsSectionSubtitle}>
                      Select which vision model is currently active for photo analysis & multimodal recognition.
                    </Text>

                    {AVAILABLE_MODELS.filter(m => m.category === 'vision' && downloadedModels[m.id]).length === 0 ? (
                      <View style={styles.emptyModelCategoryCard}>
                        <Text style={styles.emptyModelCategoryText}>
                          No vision models downloaded yet.
                        </Text>
                        <Pressable
                          style={[styles.browseModelsChip, { borderColor: '#DDD6FE', backgroundColor: '#F5F3FF' }]}
                          onPress={() => {
                            setModelCategoryFilter('vision');
                            setSettingsTab('models');
                          }}>
                          <Plus size={12} color="#7C3AED" />
                          <Text style={[styles.browseModelsChipText, { color: '#7C3AED' }]}>Download Vision Model</Text>
                        </Pressable>
                      </View>
                    ) : (
                      <View style={{ gap: 8, marginTop: 4 }}>
                        {AVAILABLE_MODELS.filter(m => m.category === 'vision' && downloadedModels[m.id]).map(m => {
                          const isCurrentlyRunning = m.id === selectedModelId && phase === 'ready';
                          const isSelectedVision = m.id === activeVisionModelId;

                          return (
                            <View
                              key={m.id}
                              style={[
                                styles.downloadedItemCard,
                                isCurrentlyRunning && [styles.downloadedItemCardActive, { borderColor: '#7C3AED', backgroundColor: '#FAF5FF' }],
                              ]}>
                              <View style={styles.downloadedItemTop}>
                                <View style={{ flex: 1, marginRight: 8 }}>
                                  <Text style={[styles.downloadedItemName, isCurrentlyRunning && { color: '#7C3AED' }]}>
                                    {m.name}
                                  </Text>
                                  <Text style={styles.downloadedItemDesc} numberOfLines={2}>
                                    {m.description}
                                  </Text>
                                </View>
                                <View style={[styles.modelTag, { backgroundColor: '#F3E8FF' }]}>
                                  <Text style={[styles.modelTagText, { color: '#7C3AED' }]}>
                                    {formatSize(m.size + (m.mmprojSize || 0))}
                                  </Text>
                                </View>
                              </View>

                              <View style={styles.downloadedItemBottom}>
                                {isCurrentlyRunning ? (
                                  <View style={styles.modelActiveBadge}>
                                    <Check size={12} color="#059669" />
                                    <Text style={styles.modelActiveBadgeText}>Active in Chat</Text>
                                  </View>
                                ) : isSelectedVision ? (
                                  <View style={[styles.modelSelectedBadge, { backgroundColor: '#F3E8FF' }]}>
                                    <Check size={12} color="#7C3AED" />
                                    <Text style={[styles.modelSelectedBadgeText, { color: '#7C3AED' }]}>Selected for Vision</Text>
                                  </View>
                                ) : (
                                  <View style={styles.modelCachedBadge}>
                                    <HardDrive size={11} color={C.textSecondary} />
                                    <Text style={styles.modelCachedBadgeText}>Ready Offline</Text>
                                  </View>
                                )}

                                <View style={{ flexDirection: 'row', alignItems: 'center', gap: 6 }}>
                                  {!isCurrentlyRunning && (
                                    <Pressable
                                      style={({ pressed }) => [
                                        styles.activateBtn,
                                        { backgroundColor: '#7C3AED' },
                                        pressed && styles.pressed,
                                      ]}
                                      onPress={() => activateModel(m.id, 'vision')}>
                                      <Zap size={11} color="#FFFFFF" />
                                      <Text style={styles.activateBtnText}>Activate</Text>
                                    </Pressable>
                                  )}
                                  <Pressable
                                    style={({ pressed }) => [
                                      styles.modelDeleteBtn,
                                      pressed && styles.pressed,
                                    ]}
                                    onPress={() => deleteModel(m)}
                                    hitSlop={6}
                                    accessibilityLabel={`Delete ${m.name}`}>
                                    <Trash2 size={13} color={C.danger} />
                                  </Pressable>
                                </View>
                              </View>
                            </View>
                          );
                        })}
                      </View>
                    )}
                  </View>
                </View>
              ) : (
                /* Tab 2: Persona & Instructions */
                <View>
                  <View style={styles.settingLabelRow}>
                    <Text style={styles.settingLabel}>Custom Instructions & Persona</Text>
                    <Pressable
                      style={styles.dismissKeyboardChip}
                      onPress={() => Keyboard.dismiss()}
                      hitSlop={8}>
                      <ChevronDown size={12} color={C.accent} />
                      <Text style={styles.dismissKeyboardChipText}>Hide Keyboard</Text>
                    </Pressable>
                  </View>
                  <Text style={styles.settingHint}>
                    Tell the AI how to answer (e.g. short responses, bullet points, tone):
                  </Text>

                  <TextInput
                    style={styles.systemInput}
                    value={tempPrompt}
                    onChangeText={setTempPrompt}
                    multiline
                    placeholder="Type your custom instructions here..."
                    placeholderTextColor={C.muted}
                  />

                  <Text style={styles.presetTitle}>Quick Presets</Text>
                  <View style={styles.presetRow}>
                    <Pressable
                      style={styles.presetChip}
                      onPress={() =>
                        setTempPrompt(
                          'Always give very short, concise responses. Use brief bullet points. Avoid filler.',
                        )
                      }>
                      <Zap size={13} color={C.accent} />
                      <Text style={styles.presetChipText}>Short & Concise</Text>
                    </Pressable>
                    <Pressable
                      style={styles.presetChip}
                      onPress={() =>
                        setTempPrompt(
                          'You are an expert developer. Provide clean code snippets and brief, direct explanations.',
                        )
                      }>
                      <Code2 size={13} color={C.accent} />
                      <Text style={styles.presetChipText}>Code Expert</Text>
                    </Pressable>
                    <Pressable
                      style={styles.presetChip}
                      onPress={() =>
                        setTempPrompt(
                          'You are a helpful, concise assistant running fully offline on a phone. Answer clearly.',
                        )
                      }>
                      <Brain size={13} color={C.accent} />
                      <Text style={styles.presetChipText}>Default</Text>
                    </Pressable>
                  </View>

                  {/* Voice Responses Option */}
                  <View style={styles.voiceSettingCard}>
                    <View style={{ flex: 1, paddingRight: 10 }}>
                      <View style={{ flexDirection: 'row', alignItems: 'center', gap: 6 }}>
                        <Volume2 size={16} color={C.accent} />
                        <Text style={styles.voiceSettingTitle}>Voice Responses (Read Aloud)</Text>
                      </View>
                      <Text style={styles.voiceSettingDesc}>
                        Automatically speak AI answers out loud offline using on-device text-to-speech.
                      </Text>
                    </View>
                    <Switch
                      value={voiceResponseEnabled}
                      onValueChange={toggleVoiceResponseMode}
                      trackColor={{ false: C.border, true: C.accent }}
                      thumbColor="#FFFFFF"
                    />
                  </View>

                  <View style={styles.modalActions}>
                    <Pressable
                      style={[styles.modalBtn, styles.clearBtn]}
                      onPress={() => {
                        Keyboard.dismiss();
                        setMessages([]);
                        setSettingsOpen(false);
                      }}>
                      <Trash2 size={15} color={C.danger} />
                      <Text style={styles.clearBtnText}>Clear Chat</Text>
                    </Pressable>
                    <Pressable
                      style={[styles.modalBtn, styles.saveBtn]}
                      onPress={saveSettings}>
                      <Check size={16} color="#FFFFFF" />
                      <Text style={styles.saveBtnText}>Save Persona</Text>
                    </Pressable>
                  </View>
                </View>
              )}
            </ScrollView>
          </View>
        </KeyboardAvoidingView>
      </Modal>

      {/* Sessions / Chat History Drawer (Full Screen Pure White) */}
      <Modal
        visible={drawerOpen}
        animationType="slide"
        transparent={false}
        onRequestClose={() => setDrawerOpen(false)}>
        <StatusBar barStyle="dark-content" />
        <View style={styles.drawerOverlay}>
          <View style={styles.drawerContainer}>
            {/* Header: App Title & Close button */}
            <View style={styles.drawerHeader}>
              <View style={styles.drawerBrandRow}>
                <View style={styles.drawerLogoIcon}>
                  <Sparkles size={16} color={C.accent} />
                </View>
                <Text style={styles.drawerBrandText}>
                  <Text style={{ color: C.accent }}>Ilyas</Text>
                  <Text style={{ color: '#7C3AED' }}>Ka</Text>
                  <Text style={{ color: '#EC4899' }}>Gpt</Text>
                </Text>
              </View>
              <Pressable
                style={styles.drawerCloseBtn}
                onPress={() => setDrawerOpen(false)}
                hitSlop={8}>
                <X size={18} color={C.muted} />
              </Pressable>
            </View>

            {/* + New Chat Button (Pill button like Image 2) */}
            <Pressable
              style={({ pressed }) => [
                styles.drawerNewChatBtn,
                pressed && styles.pressed,
              ]}
              onPress={startNewChat}>
              <Plus size={18} color={C.accent} />
              <Text style={styles.drawerNewChatText}>New chat</Text>
            </Pressable>

            {/* Sessions List */}
            <ScrollView
              style={styles.drawerScroll}
              showsVerticalScrollIndicator={false}>
              {todaySessions.length > 0 && (
                <View style={styles.drawerSection}>
                  <Text style={styles.drawerSectionHeader}>TODAY</Text>
                  {todaySessions.map(s => {
                    const isCurrent = s.id === currentSessionId;
                    return (
                      <Pressable
                        key={s.id}
                        style={({ pressed }) => [
                          styles.sessionItem,
                          isCurrent && styles.sessionItemActive,
                          pressed && styles.pressed,
                        ]}
                        onPress={() => selectSession(s)}>
                        <MessageSquare
                          size={15}
                          color={isCurrent ? C.accent : C.muted}
                          style={{ marginRight: 10 }}
                        />
                        <Text
                          style={[
                            styles.sessionTitle,
                            isCurrent && styles.sessionTitleActive,
                          ]}
                          numberOfLines={1}>
                          {s.title || 'Conversation'}
                        </Text>
                        <Pressable
                          style={styles.sessionDeleteBtn}
                          onPress={() => deleteSession(s.id, s.title)}
                          hitSlop={8}>
                          <Trash2 size={13} color={isCurrent ? C.accent : C.muted} />
                        </Pressable>
                      </Pressable>
                    );
                  })}
                </View>
              )}

              {previous7DaysSessions.length > 0 && (
                <View style={styles.drawerSection}>
                  <Text style={styles.drawerSectionHeader}>PREVIOUS 7 DAYS</Text>
                  {previous7DaysSessions.map(s => {
                    const isCurrent = s.id === currentSessionId;
                    return (
                      <Pressable
                        key={s.id}
                        style={({ pressed }) => [
                          styles.sessionItem,
                          isCurrent && styles.sessionItemActive,
                          pressed && styles.pressed,
                        ]}
                        onPress={() => selectSession(s)}>
                        <MessageSquare
                          size={15}
                          color={isCurrent ? C.accent : C.muted}
                          style={{ marginRight: 10 }}
                        />
                        <Text
                          style={[
                            styles.sessionTitle,
                            isCurrent && styles.sessionTitleActive,
                          ]}
                          numberOfLines={1}>
                          {s.title || 'Conversation'}
                        </Text>
                        <Pressable
                          style={styles.sessionDeleteBtn}
                          onPress={() => deleteSession(s.id, s.title)}
                          hitSlop={8}>
                          <Trash2 size={13} color={isCurrent ? C.accent : C.muted} />
                        </Pressable>
                      </Pressable>
                    );
                  })}
                </View>
              )}

              {olderSessions.length > 0 && (
                <View style={styles.drawerSection}>
                  <Text style={styles.drawerSectionHeader}>OLDER</Text>
                  {olderSessions.map(s => {
                    const isCurrent = s.id === currentSessionId;
                    return (
                      <Pressable
                        key={s.id}
                        style={({ pressed }) => [
                          styles.sessionItem,
                          isCurrent && styles.sessionItemActive,
                          pressed && styles.pressed,
                        ]}
                        onPress={() => selectSession(s)}>
                        <MessageSquare
                          size={15}
                          color={isCurrent ? C.accent : C.muted}
                          style={{ marginRight: 10 }}
                        />
                        <Text
                          style={[
                            styles.sessionTitle,
                            isCurrent && styles.sessionTitleActive,
                          ]}
                          numberOfLines={1}>
                          {s.title || 'Conversation'}
                        </Text>
                        <Pressable
                          style={styles.sessionDeleteBtn}
                          onPress={() => deleteSession(s.id, s.title)}
                          hitSlop={8}>
                          <Trash2 size={13} color={isCurrent ? C.accent : C.muted} />
                        </Pressable>
                      </Pressable>
                    );
                  })}
                </View>
              )}

              {sessions.length === 0 && (
                <View style={styles.drawerEmpty}>
                  <Text style={styles.drawerEmptyText}>No conversations yet</Text>
                </View>
              )}
            </ScrollView>

            {/* Bottom Actions Bar (Model Center, Settings, Clear All) */}
            <View style={styles.drawerFooter}>
              <Pressable
                style={({ pressed }) => [
                  styles.drawerFooterItem,
                  pressed && styles.pressed,
                ]}
                onPress={() => {
                  setDrawerOpen(false);
                  setSettingsTab('models');
                  setSettingsOpen(true);
                }}>
                <Cpu size={18} color={C.accent} />
                <View style={{ flex: 1 }}>
                  <Text style={styles.drawerFooterLabel}>Model Center</Text>
                  <Text style={styles.drawerFooterSub}>
                    {activeModel.name.split(' ')[1] || activeModel.name} · {formatSize(activeModel.size)}
                  </Text>
                </View>
                <ChevronRight size={16} color={C.muted} />
              </Pressable>

              <Pressable
                style={({ pressed }) => [
                  styles.drawerFooterItem,
                  pressed && styles.pressed,
                ]}
                onPress={() => {
                  setDrawerOpen(false);
                  setSettingsTab('downloads');
                  setSettingsOpen(true);
                  refreshDownloadedStatus();
                }}>
                <HardDrive size={18} color={C.accent} />
                <View style={{ flex: 1 }}>
                  <Text style={styles.drawerFooterLabel}>Downloads & Active Brains</Text>
                  <Text style={styles.drawerFooterSub}>
                    {(isDownloadingRef.current || downloadingModelId !== null)
                      ? `Downloading ${(progress * 100).toFixed(0)}%`
                      : 'Manage text & vision models'}
                  </Text>
                </View>
                <ChevronRight size={16} color={C.muted} />
              </Pressable>

              <Pressable
                style={({ pressed }) => [
                  styles.drawerFooterItem,
                  pressed && styles.pressed,
                ]}
                onPress={() => {
                  setDrawerOpen(false);
                  setTempPrompt(systemPrompt);
                  setSettingsTab('persona');
                  setSettingsOpen(true);
                }}>
                <Settings size={18} color={C.textSecondary} />
                <Text style={styles.drawerFooterLabel}>Settings & Persona</Text>
                <ChevronRight size={16} color={C.muted} />
              </Pressable>

              <Pressable
                style={({ pressed }) => [
                  styles.drawerFooterItem,
                  pressed && styles.pressed,
                ]}
                onPress={clearAllSessions}>
                <Trash2 size={16} color={C.danger} />
                <Text style={[styles.drawerFooterLabel, { color: C.danger }]}>
                  Clear All History
                </Text>
              </Pressable>
            </View>
          </View>
        </View>
      </Modal>
    </KeyboardAvoidingView>
  );
}

// ---- Markdown Code Block & Message Renderer -------------------------------
function MessageContent({
  content,
  role,
  isMediaUser,
}: {
  content: string;
  role: Role;
  isMediaUser?: boolean;
}) {
  if (role === 'user' || !content.includes('```')) {
    return (
      <Text
        style={
          isMediaUser
            ? styles.userMediaText
            : role === 'user'
            ? styles.userText
            : styles.botText
        }
        selectable>
        {content || '…'}
      </Text>
    );
  }

  const parts = content.split(/(```[\s\S]*?```)/g);
  return (
    <View style={styles.messageCol}>
      {parts.map((part, index) => {
        if (part.startsWith('```') && part.endsWith('```')) {
          const body = part.slice(3, -3).trim();
          const newlineIdx = body.indexOf('\n');
          let lang = 'code';
          let codeText = body;
          if (newlineIdx !== -1) {
            const firstLine = body.slice(0, newlineIdx).trim();
            if (/^[a-zA-Z0-9_#+-]+$/.test(firstLine)) {
              lang = firstLine;
              codeText = body.slice(newlineIdx + 1);
            }
          }
          return <CodeCard key={index} code={codeText} lang={lang} />;
        }
        if (!part.trim()) {
          return null;
        }
        return (
          <Text key={index} style={styles.botText} selectable>
            {part.trim()}
          </Text>
        );
      })}
    </View>
  );
}

function CodeCard({ code, lang }: { code: string; lang: string }) {
  const [copied, setCopied] = useState(false);

  const copyCode = async () => {
    try {
      setCopied(true);
      await Share.share({ message: code });
      setTimeout(() => setCopied(false), 2000);
    } catch {
      setCopied(false);
    }
  };

  return (
    <View style={styles.codeCard}>
      <View style={styles.codeHeader}>
        <View style={styles.codeHeaderLeft}>
          <Code2 size={13} color={C.codeLang} />
          <Text style={styles.codeLang}>{lang.toUpperCase()}</Text>
        </View>
        <Pressable
          style={({ pressed }) => [styles.copyBtn, pressed && styles.pressed]}
          onPress={copyCode}>
          {copied ? (
            <View style={styles.copyBtnRow}>
              <Check size={12} color="#A6E3A1" />
              <Text style={[styles.copyText, { color: '#A6E3A1' }]}>Copied</Text>
            </View>
          ) : (
            <View style={styles.copyBtnRow}>
              <Copy size={12} color="#FFFFFF" />
              <Text style={styles.copyText}>Copy</Text>
            </View>
          )}
        </Pressable>
      </View>
      <ScrollView
        horizontal
        showsHorizontalScrollIndicator={false}
        nestedScrollEnabled={true}
        style={styles.codeScrollHorizontal}>
        <Text style={styles.codeContent}>
          {code}
        </Text>
      </ScrollView>
    </View>
  );
}

function Status({ text }: { text: string }) {
  return (
    <View style={styles.status}>
      <ActivityIndicator color={C.accent} />
      <Text style={styles.cardText}>{text}</Text>
    </View>
  );
}

// ---- Stylesheet (Modern Apple Light Theme Option 2) -----------------------
const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: C.bg },
  center: {
    flex: 1,
    backgroundColor: C.bg,
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 24,
    paddingTop: Platform.OS === 'ios' ? 52 : 20,
    paddingBottom: Platform.OS === 'ios' ? 34 : 20,
  },
  logo: { color: C.text, fontSize: 32, fontWeight: '800', letterSpacing: 0.5 },
  subtitle: { color: C.textSecondary, marginTop: 6, marginBottom: 28, fontSize: 15 },
  status: { flexDirection: 'row', alignItems: 'center', gap: 10 },
  card: {
    width: '100%',
    backgroundColor: C.surface,
    borderColor: C.border,
    borderWidth: 1,
    borderRadius: 20,
    padding: 20,
    gap: 12,
    shadowColor: '#000',
    shadowOpacity: 0.05,
    shadowOffset: { width: 0, height: 2 },
    shadowRadius: 8,
    elevation: 2,
  },
  cardTitle: { color: C.text, fontSize: 16, fontWeight: '700' },
  cardText: { color: C.text, fontSize: 14 },
  hint: { color: C.muted, fontSize: 13, lineHeight: 18 },
  barTrack: {
    height: 10,
    borderRadius: 5,
    backgroundColor: C.surface2,
    overflow: 'hidden',
  },
  barFill: { height: '100%', borderRadius: 5, backgroundColor: C.accent },
  progressStatsRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginTop: 8,
    marginBottom: 4,
  },
  progressPercentText: {
    fontSize: 15,
    fontWeight: '800',
    color: C.accent,
  },
  progressBytesText: {
    fontSize: 13,
    fontWeight: '700',
    color: C.text,
  },
  speedEtaRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    marginBottom: 6,
  },
  speedBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    backgroundColor: C.accentSoft,
    paddingHorizontal: 8,
    paddingVertical: 3,
    borderRadius: 8,
  },
  speedBadgeText: {
    fontSize: 11,
    fontWeight: '700',
    color: C.accent,
  },
  etaText: {
    fontSize: 12,
    fontWeight: '600',
    color: C.textSecondary,
  },
  button: {
    backgroundColor: C.accent,
    borderRadius: 12,
    paddingVertical: 12,
    alignItems: 'center',
  },
  buttonText: { color: '#fff', fontWeight: '700' },
  pressed: { opacity: 0.7, transform: [{ scale: 0.98 }] },
  disabled: { opacity: 0.4 },

  // Edge-to-Edge Navigation Header
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 16,
    paddingTop: Platform.OS === 'ios' ? 52 : 16,
    paddingBottom: 12,
    borderBottomWidth: 1,
    borderBottomColor: C.border,
    backgroundColor: C.surface,
  },
  headerLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
    flex: 1,
    marginRight: 6,
  },
  headerBrandStack: {
    flexDirection: 'column',
    justifyContent: 'center',
    flexShrink: 1,
  },
  headerBrandRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
  },
  headerLogoBadge: {
    width: 20,
    height: 20,
    borderRadius: 6,
    backgroundColor: '#F3E8FF',
    alignItems: 'center',
    justifyContent: 'center',
  },
  headerTitle: {
    fontSize: 18,
    fontWeight: '800',
    letterSpacing: -0.3,
  },
  headerSubtitleRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 5,
    marginTop: 2,
  },
  statusDot: {
    width: 6,
    height: 6,
    borderRadius: 3,
    backgroundColor: '#10B981',
  },
  headerSubtitleText: {
    color: C.textSecondary,
    fontSize: 11,
    fontWeight: '600',
    letterSpacing: -0.1,
    maxWidth: 140,
  },
  headerRight: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 7,
  },
  headerIconBtn: {
    width: 36,
    height: 36,
    borderRadius: 18,
    backgroundColor: C.surface2,
    borderWidth: 1,
    borderColor: C.border,
    alignItems: 'center',
    justifyContent: 'center',
  },
  headerVoiceBtnActive: {
    backgroundColor: C.accentSoft,
    borderColor: C.accent,
  },

  // Chat message list
  list: {
    paddingHorizontal: 16,
    paddingTop: 16,
    paddingBottom: 16,
    gap: 14,
    flexGrow: 1,
  },
  emptyCard: {
    backgroundColor: C.surface,
    borderRadius: 24,
    padding: 20,
    marginTop: 8,
    borderWidth: 1,
    borderColor: C.border,
    shadowColor: '#000',
    shadowOpacity: 0.04,
    shadowOffset: { width: 0, height: 2 },
    shadowRadius: 10,
    elevation: 2,
  },
  emptyTitle: { color: C.text, fontSize: 18, fontWeight: '800', marginBottom: 4 },
  emptySubtitle: { color: C.textSecondary, fontSize: 13, marginBottom: 16 },
  chipsGrid: { flexDirection: 'row', flexWrap: 'wrap', gap: 10 },
  suggestionChip: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    backgroundColor: C.surface2,
    borderWidth: 1,
    borderColor: C.border,
    paddingHorizontal: 14,
    paddingVertical: 10,
    borderRadius: 14,
  },
  chipText: { color: C.text, fontSize: 13, fontWeight: '600' },

  // Message Bubbles
  bubble: {
    maxWidth: '88%',
    borderRadius: 20,
    paddingHorizontal: 16,
    paddingVertical: 12,
  },
  userBubble: {
    alignSelf: 'flex-end',
    backgroundColor: C.accent,
    borderBottomRightRadius: 4,
    shadowColor: C.accent,
    shadowOpacity: 0.25,
    shadowOffset: { width: 0, height: 3 },
    shadowRadius: 6,
    elevation: 3,
  },
  userMediaBubble: {
    alignSelf: 'flex-end',
    backgroundColor: '#FFFFFF',
    borderWidth: 1,
    borderColor: C.border,
    borderRadius: 20,
    borderBottomRightRadius: 4,
    padding: 8,
    maxWidth: '88%',
    shadowColor: '#000',
    shadowOpacity: 0.06,
    shadowOffset: { width: 0, height: 2 },
    shadowRadius: 8,
    elevation: 2,
  },
  userMediaText: {
    color: C.text,
    fontSize: 15,
    lineHeight: 22,
    marginTop: 8,
    paddingHorizontal: 4,
  },
  botBubble: {
    alignSelf: 'flex-start',
    backgroundColor: C.surface,
    borderWidth: 1,
    borderColor: C.border,
    borderBottomLeftRadius: 4,
    shadowColor: '#000',
    shadowOpacity: 0.04,
    shadowOffset: { width: 0, height: 2 },
    shadowRadius: 8,
    elevation: 1,
  },
  botHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 6,
  },
  botHeaderLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
  },
  botSpeakBtn: {
    padding: 5,
    borderRadius: 8,
    backgroundColor: C.surface2,
    borderWidth: 1,
    borderColor: C.border,
  },
  botSpeakBtnActive: {
    backgroundColor: '#FEE2E2',
    borderColor: '#FCA5A5',
  },
  voiceSettingCard: {
    backgroundColor: C.surface2,
    borderRadius: 16,
    padding: 14,
    borderWidth: 1,
    borderColor: C.border,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 14,
  },
  voiceSettingTitle: {
    fontSize: 14,
    fontWeight: '700',
    color: C.text,
  },
  voiceSettingDesc: {
    fontSize: 12,
    color: C.textSecondary,
    marginTop: 3,
    lineHeight: 16,
  },
  floatingScrollBtn: {
    position: 'absolute',
    bottom: 82,
    right: 20,
    backgroundColor: C.surface,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    paddingHorizontal: 12,
    paddingVertical: 7,
    borderRadius: 20,
    borderWidth: 1,
    borderColor: C.border,
    shadowColor: '#000',
    shadowOpacity: 0.12,
    shadowOffset: { width: 0, height: 3 },
    shadowRadius: 6,
    elevation: 4,
    zIndex: 99,
  },
  floatingScrollText: {
    fontSize: 12,
    fontWeight: '700',
    color: C.accent,
  },
  botBadge: {
    width: 18,
    height: 18,
    borderRadius: 9,
    backgroundColor: C.accentSoft,
    alignItems: 'center',
    justifyContent: 'center',
  },
  botName: { fontSize: 12, fontWeight: '700', color: C.muted },
  userText: { color: '#FFFFFF', fontSize: 15, lineHeight: 22 },
  botText: { color: C.text, fontSize: 15, lineHeight: 22 },
  messageCol: { gap: 8 },

  // Bubble Attachment Previews
  bubbleAttachmentBox: { marginBottom: 8 },
  bubbleImage: {
    width: 220,
    height: 160,
    borderRadius: 14,
    backgroundColor: C.border,
  },
  bubbleDocCard: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
    backgroundColor: C.surface2,
    padding: 10,
    borderRadius: 14,
    borderWidth: 1,
    borderColor: C.border,
  },
  bubbleDocCardUser: {
    backgroundColor: 'rgba(255, 255, 255, 0.18)',
    borderColor: 'rgba(255, 255, 255, 0.3)',
  },
  bubbleDocIcon: {
    width: 36,
    height: 36,
    borderRadius: 10,
    backgroundColor: C.accentSoft,
    alignItems: 'center',
    justifyContent: 'center',
  },
  bubbleDocIconUser: {
    backgroundColor: 'rgba(255, 255, 255, 0.25)',
  },
  bubbleDocName: { fontSize: 13, fontWeight: '700', color: C.text },
  bubbleDocMeta: { fontSize: 11, color: C.textSecondary, marginTop: 2 },

  // Code Block in Dark Obsidian Theme with Copy
  codeCard: {
    backgroundColor: C.codeBg,
    borderRadius: 14,
    borderWidth: 1,
    borderColor: C.codeBorder,
    overflow: 'hidden',
    marginTop: 8,
    marginBottom: 4,
    width: '100%',
    maxHeight: 400,
  },
  codeScrollHorizontal: {
    maxHeight: 340,
  },
  codeHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    backgroundColor: C.codeHeader,
    paddingHorizontal: 12,
    paddingVertical: 7,
    borderBottomWidth: 1,
    borderBottomColor: C.codeBorder,
  },
  codeHeaderLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
  },
  codeLang: {
    color: C.codeLang,
    fontSize: 11,
    fontWeight: '800',
    letterSpacing: 0.5,
  },
  copyBtn: {
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 6,
    backgroundColor: 'rgba(255,255,255,0.08)',
  },
  copyBtnRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
  },
  copyText: { color: '#FFFFFF', fontSize: 11, fontWeight: '600' },
  codeContent: {
    fontFamily: Platform.OS === 'ios' ? 'Menlo' : 'monospace',
    color: C.codeText,
    fontSize: 13,
    lineHeight: 19,
    padding: 12,
  },

  speed: { color: C.muted, fontSize: 11, textAlign: 'center', marginBottom: 4 },

  // Bottom Input Bar
  inputContainer: {
    paddingHorizontal: 16,
    paddingTop: 8,
    paddingBottom: Platform.OS === 'ios' ? 34 : 14,
    backgroundColor: C.bg,
    zIndex: 10,
    elevation: 10,
  },
  micBtn: {
    width: 32,
    height: 32,
    borderRadius: 16,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: C.surface2,
  },
  micBtnActive: {
    backgroundColor: C.danger,
  },
  recordingBar: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#FEF2F2',
    borderWidth: 1,
    borderColor: '#FCA5A5',
    borderRadius: 14,
    paddingHorizontal: 12,
    paddingVertical: 6,
    marginBottom: 8,
    gap: 8,
  },
  recordingPulse: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: C.danger,
  },
  recordingText: {
    color: C.danger,
    fontSize: 12,
    fontWeight: '700',
    flex: 1,
  },
  modelHubHeader: {
    marginTop: 18,
    marginBottom: 4,
  },
  modelSectionTitle: {
    color: C.text,
    fontSize: 15,
    fontWeight: '700',
    marginBottom: 2,
  },
  categoryTabRow: {
    flexDirection: 'row',
    backgroundColor: C.surface2,
    borderRadius: 12,
    padding: 3,
    marginTop: 10,
    marginBottom: 12,
    gap: 4,
    borderWidth: 1,
    borderColor: C.border,
  },
  categoryTabBtn: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 6,
    paddingVertical: 8,
    borderRadius: 9,
  },
  categoryTabBtnActive: {
    backgroundColor: C.accent,
  },
  categoryTabBtnText: {
    fontSize: 12,
    fontWeight: '700',
    color: C.textSecondary,
  },
  categoryTabBtnTextActive: {
    color: '#FFFFFF',
  },
  modelCard: {
    backgroundColor: C.surface2,
    borderRadius: 16,
    borderWidth: 1.5,
    borderColor: C.border,
    padding: 14,
    marginTop: 4,
    gap: 8,
  },
  modelCardActive: {
    borderColor: C.accent,
    backgroundColor: C.accentSoft,
  },
  modelCardTop: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  modelCardName: {
    fontSize: 14,
    fontWeight: '800',
    color: C.text,
  },
  modelTag: {
    paddingHorizontal: 8,
    paddingVertical: 3,
    borderRadius: 8,
    backgroundColor: C.surface,
    borderWidth: 1,
    borderColor: C.border,
  },
  modelTagActive: {
    backgroundColor: C.accent,
    borderColor: C.accent,
  },
  modelTagText: {
    fontSize: 11,
    fontWeight: '700',
    color: C.textSecondary,
  },
  modelCardDesc: {
    fontSize: 12,
    color: C.textSecondary,
    lineHeight: 16,
  },
  modelActionRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginTop: 2,
    paddingTop: 10,
    borderTopWidth: 1,
    borderTopColor: C.border,
  },
  modelActiveBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    backgroundColor: '#ECFDF5',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 8,
  },
  modelActiveBadgeText: {
    fontSize: 11,
    fontWeight: '700',
    color: '#059669',
  },
  modelDownloadedBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    backgroundColor: '#EEF2FF',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 8,
  },
  modelDownloadedBadgeText: {
    fontSize: 11,
    fontWeight: '700',
    color: C.accent,
  },
  modelNotDownloadedBadge: {
    paddingHorizontal: 6,
    paddingVertical: 4,
  },
  modelNotDownloadedText: {
    fontSize: 11,
    fontWeight: '600',
    color: C.muted,
  },
  modelBtnGroup: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  modelSwitchBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    backgroundColor: C.accent,
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 12,
  },
  modelSwitchBtnText: {
    color: '#FFFFFF',
    fontSize: 12,
    fontWeight: '700',
  },
  modelDeleteBtn: {
    padding: 6,
    borderRadius: 12,
    backgroundColor: '#FEF2F2',
    borderWidth: 1,
    borderColor: '#FEE2E2',
    alignItems: 'center',
    justifyContent: 'center',
  },
  modelDownloadBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 5,
    backgroundColor: C.surface,
    borderWidth: 1.5,
    borderColor: C.accent,
    paddingHorizontal: 10,
    paddingVertical: 5,
    borderRadius: 12,
  },
  modelDownloadBtnText: {
    color: C.accent,
    fontSize: 12,
    fontWeight: '700',
  },
  cancelDownloadBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 6,
    marginTop: 14,
    paddingVertical: 8,
    paddingHorizontal: 14,
    borderRadius: 12,
    backgroundColor: '#FEF2F2',
    borderWidth: 1,
    borderColor: '#FEE2E2',
  },
  cancelDownloadText: {
    color: C.danger,
    fontSize: 13,
    fontWeight: '700',
  },
  inputCard: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    backgroundColor: C.surface,
    borderWidth: 1,
    borderColor: C.border,
    borderRadius: 26,
    paddingHorizontal: 16,
    paddingVertical: 6,
    shadowColor: '#000',
    shadowOpacity: 0.05,
    shadowOffset: { width: 0, height: 2 },
    shadowRadius: 8,
    elevation: 3,
  },
  input: {
    flex: 1,
    minHeight: 40,
    maxHeight: 120,
    color: C.text,
    fontSize: 15,
    paddingVertical: 6,
  },
  sendBtn: {
    width: 38,
    height: 38,
    borderRadius: 19,
    backgroundColor: C.accent,
    alignItems: 'center',
    justifyContent: 'center',
  },
  stopBtn: { backgroundColor: C.danger },
  sendText: { color: '#fff', fontSize: 17, fontWeight: '800' },

  // Attachment Preview & Buttons
  previewBar: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: C.surface,
    borderRadius: 16,
    padding: 8,
    marginBottom: 8,
    borderWidth: 1,
    borderColor: C.border,
    gap: 10,
    shadowColor: '#000',
    shadowOpacity: 0.04,
    shadowOffset: { width: 0, height: 2 },
    shadowRadius: 6,
    elevation: 2,
  },
  previewThumb: {
    width: 38,
    height: 38,
    borderRadius: 8,
    backgroundColor: C.border,
  },
  previewDocIcon: {
    width: 38,
    height: 38,
    borderRadius: 8,
    backgroundColor: C.accentSoft,
    alignItems: 'center',
    justifyContent: 'center',
  },
  previewInfo: { flex: 1 },
  previewName: { fontSize: 13, fontWeight: '700', color: C.text },
  previewSub: { fontSize: 11, color: C.muted, marginTop: 1 },
  previewCloseBtn: {
    padding: 6,
    borderRadius: 12,
    backgroundColor: C.surface2,
  },
  quickVisionBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    backgroundColor: '#7C3AED',
    paddingHorizontal: 9,
    paddingVertical: 5,
    borderRadius: 8,
    marginRight: 6,
  },
  quickVisionBtnText: {
    color: '#FFFFFF',
    fontSize: 11,
    fontWeight: '700',
  },
  quickVisionBtnOutline: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    backgroundColor: C.accentSoft,
    borderWidth: 1,
    borderColor: C.accent,
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 8,
    marginRight: 6,
  },
  quickVisionBtnOutlineText: {
    color: C.accent,
    fontSize: 11,
    fontWeight: '700',
  },
  attachBtn: {
    width: 32,
    height: 32,
    borderRadius: 16,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: C.surface2,
  },

  // Attachment Modal / Bottom Sheet
  attachOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.4)',
    justifyContent: 'flex-end',
  },
  attachSheet: {
    backgroundColor: C.surface,
    borderTopLeftRadius: 26,
    borderTopRightRadius: 26,
    paddingHorizontal: 20,
    paddingTop: 18,
    paddingBottom: Platform.OS === 'ios' ? 36 : 22,
    shadowColor: '#000',
    shadowOpacity: 0.1,
    shadowOffset: { width: 0, height: -3 },
    shadowRadius: 12,
    elevation: 8,
  },
  attachHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingBottom: 14,
    borderBottomWidth: 1,
    borderBottomColor: C.border,
  },
  attachTitle: { fontSize: 17, fontWeight: '800', color: C.text },
  attachOptionsRow: { flexDirection: 'row', gap: 12, marginTop: 16 },
  attachOptionCard: {
    flex: 1,
    backgroundColor: C.surface2,
    borderRadius: 18,
    paddingVertical: 16,
    paddingHorizontal: 6,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: C.border,
  },
  attachIconCircle: {
    width: 48,
    height: 48,
    borderRadius: 24,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 8,
  },
  attachOptionTitle: {
    fontSize: 13,
    fontWeight: '700',
    color: C.text,
    textAlign: 'center',
  },
  attachOptionDesc: {
    fontSize: 11,
    color: C.muted,
    marginTop: 2,
    textAlign: 'center',
  },

  // Settings Modal
  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.5)',
    justifyContent: 'flex-end',
  },
  modalBackdrop: {
    flex: 1,
  },
  modalContent: {
    backgroundColor: C.surface,
    borderTopLeftRadius: 28,
    borderTopRightRadius: 28,
    paddingHorizontal: 22,
    paddingTop: 18,
    paddingBottom: Platform.OS === 'ios' ? 34 : 22,
    maxHeight: '85%',
    shadowColor: '#000',
    shadowOpacity: 0.1,
    shadowOffset: { width: 0, height: -4 },
    shadowRadius: 16,
    elevation: 10,
  },
  modalContentKeyboard: {
    maxHeight: '62%',
    paddingBottom: 16,
  },
  modalHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingBottom: 14,
    borderBottomWidth: 1,
    borderBottomColor: C.border,
    marginBottom: 14,
  },
  modalTitleRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  modalTitle: { color: C.text, fontSize: 18, fontWeight: '800' },
  modalHeaderActions: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
  },
  dismissKeyboardBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    backgroundColor: C.accentSoft,
    paddingHorizontal: 10,
    paddingVertical: 5,
    borderRadius: 14,
  },
  dismissKeyboardText: {
    color: C.accent,
    fontSize: 12,
    fontWeight: '700',
  },
  modalCloseBtn: {
    padding: 4,
  },
  settingLabelRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 4,
  },
  settingLabel: { color: C.text, fontSize: 15, fontWeight: '700' },
  dismissKeyboardChip: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    backgroundColor: C.surface2,
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: C.border,
  },
  dismissKeyboardChipText: {
    color: C.accent,
    fontSize: 11,
    fontWeight: '700',
  },
  settingHint: { color: C.textSecondary, fontSize: 13, marginBottom: 12 },
  systemInput: {
    backgroundColor: C.surface2,
    color: C.text,
    borderRadius: 14,
    padding: 14,
    fontSize: 14,
    lineHeight: 20,
    minHeight: 90,
    textAlignVertical: 'top',
    borderColor: C.border,
    borderWidth: 1,
  },
  presetTitle: { color: C.text, fontSize: 14, fontWeight: '700', marginTop: 16, marginBottom: 8 },
  presetRow: { flexDirection: 'row', flexWrap: 'wrap', gap: 8, marginBottom: 20 },
  presetChip: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    backgroundColor: C.surface2,
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 20,
    borderColor: C.border,
    borderWidth: 1,
  },
  presetChipText: { color: C.text, fontSize: 12, fontWeight: '600' },
  modalActions: { flexDirection: 'row', gap: 10, marginTop: 10, paddingBottom: 10 },
  modalBtn: {
    flex: 1,
    flexDirection: 'row',
    gap: 6,
    paddingVertical: 14,
    borderRadius: 14,
    alignItems: 'center',
    justifyContent: 'center',
  },
  clearBtn: { backgroundColor: C.surface2, borderWidth: 1, borderColor: C.danger },
  clearBtnText: { color: C.danger, fontWeight: '700', fontSize: 14 },
  saveBtn: { backgroundColor: C.accent },
  saveBtnText: { color: '#fff', fontWeight: '700', fontSize: 14 },

  // Tab Switcher inside Modal
  tabContainer: {
    flexDirection: 'row',
    backgroundColor: C.surface2,
    borderRadius: 14,
    padding: 3,
    marginBottom: 16,
    borderWidth: 1,
    borderColor: C.border,
  },
  tabButton: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    paddingVertical: 10,
    borderRadius: 11,
  },
  tabButtonActive: {
    backgroundColor: '#FFFFFF',
    shadowColor: '#000',
    shadowOpacity: 0.08,
    shadowOffset: { width: 0, height: 2 },
    shadowRadius: 4,
    elevation: 2,
  },
  tabButtonText: {
    fontSize: 13,
    fontWeight: '600',
    color: C.textSecondary,
  },
  tabButtonTextActive: {
    color: C.accent,
    fontWeight: '700',
  },

  // Sessions / Chat History Drawer - Full Screen Pure White
  drawerOverlay: {
    flex: 1,
    backgroundColor: '#FFFFFF',
  },
  drawerBackdrop: {
    display: 'none',
  },
  drawerContainer: {
    flex: 1,
    width: '100%',
    height: '100%',
    backgroundColor: '#FFFFFF',
    paddingTop: Platform.OS === 'ios' ? 54 : 24,
    paddingBottom: Platform.OS === 'ios' ? 34 : 20,
    paddingHorizontal: 20,
  },
  drawerHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingBottom: 16,
    borderBottomWidth: 1,
    borderBottomColor: C.border,
  },
  drawerBrandRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  drawerLogoIcon: {
    width: 28,
    height: 28,
    borderRadius: 8,
    backgroundColor: C.accentSoft,
    alignItems: 'center',
    justifyContent: 'center',
  },
  drawerBrandText: {
    fontSize: 18,
    fontWeight: '800',
    letterSpacing: -0.3,
  },
  drawerCloseBtn: {
    padding: 6,
    borderRadius: 14,
    backgroundColor: C.surface2,
  },
  drawerNewChatBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
    backgroundColor: C.surface2,
    borderWidth: 1,
    borderColor: C.border,
    borderRadius: 14,
    paddingVertical: 12,
    paddingHorizontal: 14,
    marginTop: 14,
    marginBottom: 10,
  },
  drawerNewChatText: {
    fontSize: 15,
    fontWeight: '700',
    color: C.text,
  },
  drawerScroll: {
    flex: 1,
    marginTop: 6,
  },
  drawerSection: {
    marginBottom: 18,
  },
  drawerSectionHeader: {
    fontSize: 11,
    fontWeight: '800',
    color: C.muted,
    letterSpacing: 0.6,
    marginBottom: 8,
    paddingHorizontal: 6,
  },
  sessionItem: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 10,
    paddingHorizontal: 12,
    borderRadius: 12,
    marginBottom: 3,
  },
  sessionItemActive: {
    backgroundColor: C.accentSoft,
  },
  sessionTitle: {
    flex: 1,
    fontSize: 14,
    fontWeight: '600',
    color: C.textSecondary,
  },
  sessionTitleActive: {
    color: C.accent,
    fontWeight: '700',
  },
  sessionDeleteBtn: {
    padding: 6,
    borderRadius: 8,
  },
  drawerEmpty: {
    paddingVertical: 32,
    alignItems: 'center',
  },
  drawerEmptyText: {
    fontSize: 13,
    color: C.muted,
  },
  drawerFooter: {
    paddingTop: 12,
    borderTopWidth: 1,
    borderTopColor: C.border,
    gap: 4,
  },
  drawerFooterItem: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
    paddingVertical: 10,
    paddingHorizontal: 8,
    borderRadius: 10,
  },
  drawerFooterLabel: {
    fontSize: 14,
    fontWeight: '600',
    color: C.text,
  },
  drawerFooterSub: {
    fontSize: 11,
    color: C.muted,
    marginTop: 1,
  },

  // Download Card Header & Minimize
  downloadCardHeader: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    justifyContent: 'space-between',
    marginBottom: 4,
  },
  downloadCardSubtitle: {
    fontSize: 12,
    color: C.textSecondary,
    marginTop: 2,
  },
  cardMinimizeIconBtn: {
    width: 34,
    height: 34,
    borderRadius: 17,
    backgroundColor: C.surface2,
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 1,
    borderColor: C.border,
  },
  mbDetailsRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    backgroundColor: C.surface2,
    paddingVertical: 8,
    paddingHorizontal: 12,
    borderRadius: 10,
    marginTop: 2,
    marginBottom: 4,
  },
  mbDetailsText: {
    fontSize: 13,
    color: C.textSecondary,
  },
  mbDetailsBold: {
    fontWeight: '800',
    color: C.text,
  },
  mbDetailsDivider: {
    color: C.muted,
  },
  downloadBtnGroup: {
    width: '100%',
    gap: 10,
    marginTop: 14,
  },
  minimizeDownloadBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 6,
    paddingVertical: 12,
    paddingHorizontal: 16,
    borderRadius: 12,
    backgroundColor: C.accent,
  },
  minimizeDownloadText: {
    color: '#FFFFFF',
    fontSize: 14,
    fontWeight: '700',
  },

  // Mini Download Banner under Header
  miniDownloadBanner: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 16,
    paddingVertical: 9,
    backgroundColor: '#EEF2FF',
    borderBottomWidth: 1,
    borderBottomColor: '#E0E7FF',
    gap: 10,
  },
  miniDownloadLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
    flex: 1,
  },
  miniDownloadIconBadge: {
    width: 28,
    height: 28,
    borderRadius: 14,
    backgroundColor: '#FFFFFF',
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 1,
    borderColor: '#C7D2FE',
  },
  miniDownloadTitleRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  miniDownloadTitle: {
    fontSize: 12,
    fontWeight: '700',
    color: C.text,
    flex: 1,
    marginRight: 6,
  },
  miniDownloadPercent: {
    fontSize: 12,
    fontWeight: '800',
    color: C.accent,
  },
  miniDownloadStats: {
    fontSize: 11,
    color: C.textSecondary,
    marginTop: 1,
  },
  miniDownloadActions: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
  },
  miniDownloadBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    backgroundColor: '#FFFFFF',
    borderWidth: 1,
    borderColor: '#C7D2FE',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 8,
  },
  miniDownloadBtnText: {
    fontSize: 11,
    fontWeight: '700',
    color: C.accent,
  },
  miniDownloadSettingsBtn: {
    width: 26,
    height: 26,
    borderRadius: 6,
    backgroundColor: '#FFFFFF',
    borderWidth: 1,
    borderColor: C.border,
    alignItems: 'center',
    justifyContent: 'center',
  },

  // Downloads Tab in Settings Modal
  tabActiveDot: {
    width: 6,
    height: 6,
    borderRadius: 3,
    backgroundColor: C.accent,
    marginLeft: 2,
  },
  downloadsSectionCard: {
    backgroundColor: C.surface2,
    borderRadius: 16,
    borderWidth: 1,
    borderColor: C.border,
    padding: 14,
    gap: 10,
  },
  downloadsSectionHeaderRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  downloadsSectionTitle: {
    fontSize: 14,
    fontWeight: '800',
    color: C.text,
  },
  downloadsSectionSubtitle: {
    fontSize: 12,
    color: C.textSecondary,
    lineHeight: 16,
    marginTop: -2,
    marginBottom: 4,
  },
  downloadingLiveBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    backgroundColor: '#EEF2FF',
    paddingHorizontal: 8,
    paddingVertical: 3,
    borderRadius: 8,
  },
  downloadingLiveBadgeText: {
    fontSize: 11,
    fontWeight: '700',
    color: C.accent,
  },
  dlModelName: {
    fontSize: 14,
    fontWeight: '800',
    color: C.text,
  },
  dlPartTitle: {
    fontSize: 12,
    color: C.textSecondary,
    marginTop: 2,
  },
  statsCardGrid: {
    flexDirection: 'row',
    gap: 8,
    marginTop: 4,
  },
  statBox: {
    flex: 1,
    backgroundColor: C.surface,
    borderWidth: 1,
    borderColor: C.border,
    borderRadius: 12,
    padding: 8,
    alignItems: 'center',
  },
  statBoxLabel: {
    fontSize: 10,
    fontWeight: '700',
    color: C.textSecondary,
    textTransform: 'uppercase',
    letterSpacing: 0.3,
  },
  statBoxValue: {
    fontSize: 13,
    fontWeight: '800',
    color: C.accent,
    marginTop: 2,
  },
  statBoxSub: {
    fontSize: 10,
    color: C.muted,
    marginTop: 1,
  },
  downloadsActionBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 5,
    paddingVertical: 8,
    paddingHorizontal: 12,
    borderRadius: 10,
  },
  downloadsActionBtnText: {
    fontSize: 12,
    fontWeight: '700',
  },
  noDownloadContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
    paddingVertical: 6,
  },
  noDownloadTitle: {
    fontSize: 13,
    fontWeight: '700',
    color: C.text,
  },
  noDownloadDesc: {
    fontSize: 12,
    color: C.textSecondary,
    marginTop: 1,
  },
  countBadge: {
    backgroundColor: C.accentSoft,
    paddingHorizontal: 8,
    paddingVertical: 3,
    borderRadius: 8,
  },
  countBadgeText: {
    fontSize: 11,
    fontWeight: '700',
    color: C.accent,
  },
  emptyModelCategoryCard: {
    backgroundColor: C.surface,
    borderWidth: 1,
    borderColor: C.border,
    borderStyle: 'dashed',
    borderRadius: 12,
    padding: 14,
    alignItems: 'center',
    gap: 8,
    marginTop: 4,
  },
  emptyModelCategoryText: {
    fontSize: 12,
    color: C.muted,
  },
  browseModelsChip: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 5,
    backgroundColor: C.accentSoft,
    borderWidth: 1,
    borderColor: '#C7D2FE',
    paddingHorizontal: 10,
    paddingVertical: 5,
    borderRadius: 8,
  },
  browseModelsChipText: {
    fontSize: 11,
    fontWeight: '700',
    color: C.accent,
  },
  downloadedItemCard: {
    backgroundColor: C.surface,
    borderWidth: 1,
    borderColor: C.border,
    borderRadius: 12,
    padding: 12,
    gap: 8,
  },
  downloadedItemCardActive: {
    borderColor: '#10B981',
    backgroundColor: '#F0FDF4',
  },
  downloadedItemTop: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
  },
  downloadedItemName: {
    fontSize: 13,
    fontWeight: '800',
    color: C.text,
  },
  downloadedItemDesc: {
    fontSize: 11,
    color: C.textSecondary,
    marginTop: 2,
    lineHeight: 15,
  },
  downloadedItemBottom: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingTop: 8,
    borderTopWidth: 1,
    borderTopColor: C.border,
  },
  modelSelectedBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    backgroundColor: C.accentSoft,
    paddingHorizontal: 8,
    paddingVertical: 3,
    borderRadius: 8,
  },
  modelSelectedBadgeText: {
    fontSize: 11,
    fontWeight: '700',
    color: C.accent,
  },
  modelCachedBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    paddingHorizontal: 6,
    paddingVertical: 3,
  },
  modelCachedBadgeText: {
    fontSize: 11,
    fontWeight: '600',
    color: C.textSecondary,
  },
  activateBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    backgroundColor: C.accent,
    paddingHorizontal: 10,
    paddingVertical: 5,
    borderRadius: 10,
  },
  activateBtnText: {
    color: '#FFFFFF',
    fontSize: 11,
    fontWeight: '700',
  },
  modelDownloadingBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    backgroundColor: '#EEF2FF',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 8,
  },
  modelDownloadingBadgeText: {
    fontSize: 11,
    fontWeight: '700',
    color: C.accent,
  },
});

export default App;
