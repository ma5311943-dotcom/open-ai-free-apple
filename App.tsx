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
  NativeModules,
  Platform,
  Pressable,
  SafeAreaView,
  ScrollView,
  Share,
  StatusBar,
  StyleSheet,
  Text,
  TextInput,
  TouchableWithoutFeedback,
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
export type ModelOption = {
  id: string;
  name: string;
  file: string;
  url: string;
  size: number;
  description: string;
  tag: string;
};

export const AVAILABLE_MODELS: ModelOption[] = [
  {
    id: 'qwen2.5-14b',
    name: 'Qwen2.5 14B Ultra Genius',
    file: 'Qwen2.5-14B-Instruct-Q4_K_M.gguf',
    url: 'https://huggingface.co/bartowski/Qwen2.5-14B-Instruct-GGUF/resolve/main/Qwen2.5-14B-Instruct-Q4_K_M.gguf',
    size: 8988110976,
    description: 'Ultra-intelligent 14B brain (Rivals GPT-4o-mini). Exceptional multi-step reasoning, advanced coding & all languages.',
    tag: 'Ultra Genius · 9.0 GB',
  },
  {
    id: 'qwen2.5-7b',
    name: 'Qwen2.5 7B Super Genius',
    file: 'Qwen2.5-7B-Instruct-Q4_K_M.gguf',
    url: 'https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/resolve/main/Qwen2.5-7B-Instruct-Q4_K_M.gguf',
    size: 4683074240,
    description: 'High-intelligence 7B model. Outstanding logic, programming, mathematics & multilingual fluency.',
    tag: 'Super Genius · 4.7 GB',
  },
  {
    id: 'qwen2.5-3b',
    name: 'Qwen2.5 3B Smart',
    file: 'qwen2.5-3b-instruct-q4_k_m.gguf',
    url: 'https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf',
    size: 2104932768,
    description: 'Genius Multilingual AI (Urdu, English, Hindi, Arabic, 29+ languages, Fast & Smart).',
    tag: 'Smart · 2.1 GB',
  },
  {
    id: 'qwen2.5-1.5b',
    name: 'Qwen2.5 1.5B Fast',
    file: 'qwen2.5-1.5b-instruct-q4_k_m.gguf',
    url: 'https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf',
    size: 1117320736,
    description: 'Fast & Lightweight on-device assistant. Instant responses, minimal battery & RAM usage.',
    tag: 'Fast · 1.1 GB',
  },
];

const SETTINGS_PATH = `${RNFS.DocumentDirectoryPath}/settings.json`;
const SESSIONS_PATH = `${RNFS.DocumentDirectoryPath}/chat_sessions.json`;

const SYSTEM_PROMPT =
  'You are a helpful, concise assistant running on a phone. ' +
  'Answer clearly in any language requested (English, Urdu, Hindi, Arabic, etc.). Use short code examples when useful. ' +
  'When an attached image contains extracted OCR text, accurately analyze and answer questions based on that text. ' +
  'Never hallucinate certificates, random names, or signatures not in the image.';
const STOP_WORDS = ['<|im_end|>', '<|endoftext|>', '</s>', '<|eot_id|>'];
const MAX_HISTORY = 8;
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
  } catch (err) {
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
  } catch (e) {
    // OCR fetch failed
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

function formatMB(bytes: number) {
  return formatSize(bytes);
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
  const [error, setError] = useState('');
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [generating, setGenerating] = useState(false);
  const [speed, setSpeed] = useState('');

  const [systemPrompt, setSystemPrompt] = useState(SYSTEM_PROMPT);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [settingsTab, setSettingsTab] = useState<'models' | 'persona'>('models');
  const [tempPrompt, setTempPrompt] = useState(SYSTEM_PROMPT);
  const [keyboardVisible, setKeyboardVisible] = useState(false);
  const [attachment, setAttachment] = useState<Attachment | null>(null);
  const [attachMenuOpen, setAttachMenuOpen] = useState(false);

  // Model hub: track which models are downloaded on device storage
  const [downloadedModels, setDownloadedModels] = useState<Record<string, boolean>>({});
  const downloadJobIdRef = useRef<number | null>(null);

  // Sessions / Chat History state (matching Image 2)
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [currentSessionId, setCurrentSessionId] = useState<string>('');
  const [drawerOpen, setDrawerOpen] = useState(false);

  // Voice recording state
  const [isRecording, setIsRecording] = useState(false);
  const [recordingSeconds, setRecordingSeconds] = useState(0);
  const recordingTimer = useRef<any>(null);

  const ctxRef = useRef<LlamaContext | null>(null);
  const listRef = useRef<FlatList<Message>>(null);

  const activeModel =
    AVAILABLE_MODELS.find(m => m.id === selectedModelId) || AVAILABLE_MODELS[0];

  // Refresh which models are stored locally on the phone
  const refreshDownloadedStatus = useCallback(async () => {
    const status: Record<string, boolean> = {};
    for (const m of AVAILABLE_MODELS) {
      try {
        const path = `${RNFS.DocumentDirectoryPath}/${m.file}`;
        if (await RNFS.exists(path)) {
          const stat = await RNFS.stat(path);
          status[m.id] = Number(stat.size) === m.size;
        } else {
          status[m.id] = false;
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
    if (downloadJobIdRef.current !== null) {
      try {
        RNFS.stopDownload(downloadJobIdRef.current);
      } catch {}
      downloadJobIdRef.current = null;
    }
    const model = activeModel;
    const tempPath = `${RNFS.DocumentDirectoryPath}/${model.file}.part`;
    try {
      if (await RNFS.exists(tempPath)) {
        await RNFS.unlink(tempPath);
      }
    } catch {}
    setProgress(0);
    setDownloaded(0);
    setPhase('error');
    setError('Download was canceled.');
  }, [activeModel]);

  // Model preparation and switching
  const prepareModel = useCallback(async (targetModelId?: string) => {
    const modelId = targetModelId || selectedModelId;
    const model =
      AVAILABLE_MODELS.find(m => m.id === modelId) || AVAILABLE_MODELS[0];
    const modelPath = `${RNFS.DocumentDirectoryPath}/${model.file}`;
    const tempPath = `${modelPath}.part`;

    try {
      setError('');
      setPhase('checking');

      let ready = false;
      if (await RNFS.exists(modelPath)) {
        const stat = await RNFS.stat(modelPath);
        ready = Number(stat.size) === model.size;
        if (!ready) {
          await RNFS.unlink(modelPath);
        }
      }

      if (!ready) {
        setPhase('downloading');
        setProgress(0);
        setDownloaded(0);
        if (await RNFS.exists(tempPath)) {
          await RNFS.unlink(tempPath);
        }
        const job = RNFS.downloadFile({
          fromUrl: model.url,
          toFile: tempPath,
          begin: () => {},
          progressInterval: 250,
          progress: res => {
            const total =
              res.contentLength && res.contentLength > 100000000
                ? res.contentLength
                : model.size;
            setDownloaded(res.bytesWritten);
            setProgress(Math.min(1, Math.max(0, res.bytesWritten / total)));
          },
        });
        downloadJobIdRef.current = job.jobId;
        const result = await job.promise;
        downloadJobIdRef.current = null;

        if (result.statusCode !== 200) {
          throw new Error(`Download failed (HTTP ${result.statusCode})`);
        }
        const stat = await RNFS.stat(tempPath);
        if (Number(stat.size) !== model.size) {
          throw new Error('Download incomplete. Check internet connection and retry.');
        }
        await RNFS.moveFile(tempPath, modelPath);
        await refreshDownloadedStatus();
      }

      setPhase('loading');
      if (ctxRef.current) {
        try {
          ctxRef.current.release();
        } catch {}
        ctxRef.current = null;
      }

      // For large brains like 14B Ultra Genius, optimize context size to prevent iOS RAM jetsam
      const isLargeModel = model.id === 'qwen2.5-14b' || model.size > 5000000000;
      const ctx = await initLlama({
        model: `file://${modelPath}`,
        n_ctx: isLargeModel ? 2048 : 4096,
        n_gpu_layers: 0,
        use_mlock: false,
      });
      ctxRef.current = ctx;
      setSelectedModelId(model.id);
      setPhase('ready');
      await refreshDownloadedStatus();
    } catch (e: any) {
      setError(e?.message ?? String(e));
      setPhase('error');
    }
  }, [selectedModelId, refreshDownloadedStatus]);

  // Load saved personalization and saved model choice
  useEffect(() => {
    RNFS.exists(SETTINGS_PATH).then(exists => {
      if (exists) {
        RNFS.readFile(SETTINGS_PATH, 'utf8')
          .then(raw => {
            const data = JSON.parse(raw);
            if (data.systemPrompt) {
              setSystemPrompt(data.systemPrompt);
              setTempPrompt(data.systemPrompt);
            }
            const modelToLoad = data.selectedModelId || 'qwen2.5-1.5b';
            setSelectedModelId(modelToLoad);
            prepareModel(modelToLoad);
          })
          .catch(() => prepareModel('qwen2.5-1.5b'));
      } else {
        prepareModel('qwen2.5-1.5b');
      }
    });
  }, [prepareModel]);

  const switchModel = async (newId: string) => {
    if (newId === selectedModelId && phase === 'ready') return;
    setSettingsOpen(false);
    setSelectedModelId(newId);
    try {
      await RNFS.writeFile(
        SETTINGS_PATH,
        JSON.stringify({ systemPrompt, selectedModelId: newId }),
        'utf8',
      );
    } catch {}
    prepareModel(newId);
  };

  const deleteModel = (model: ModelOption) => {
    const isCurrentlyActive = model.id === selectedModelId && phase === 'ready';
    const otherDownloaded = AVAILABLE_MODELS.find(
      m => m.id !== model.id && downloadedModels[m.id],
    );

    if (isCurrentlyActive && !otherDownloaded) {
      Alert.alert(
        'Cannot Delete Only Model',
        `This is the only downloaded model currently on your device (${formatSize(model.size)}).\n\nPlease download another model first before deleting this one so the app can continue working offline.`,
      );
      return;
    }

    Alert.alert(
      'Delete Model File',
      isCurrentlyActive && otherDownloaded
        ? `This model (${model.name}) is currently active. Deleting it will switch you to ${otherDownloaded.name} and permanently free ${formatSize(model.size)} of storage.`
        : `Delete ${model.name} from your device? This will permanently free ${formatSize(model.size)} of storage.`,
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
    if (model.size > 4000000000) {
      Alert.alert(
        `Download ${model.name}?`,
        `This is a massive ${formatSize(model.size)} AI brain.\n\n` +
          `• Connect to Wi-Fi\n` +
          `• Requires ~10 GB free phone storage\n` +
          `• Runs locally with deep reasoning\n\n` +
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
        JSON.stringify({ systemPrompt: nextPrompt, selectedModelId }),
        'utf8',
      );
    } catch {}
  }, [tempPrompt, selectedModelId, systemPrompt]);

  // Voice Recording / Dictation (100% Offline Speech Recognition)
  const toggleRecording = async () => {
    if (generating) return;

    if (isRecording) {
      // Stop recording
      if (recordingTimer.current) {
        clearInterval(recordingTimer.current);
        recordingTimer.current = null;
      }
      setIsRecording(false);

      try {
        const { NativeSpeech } = NativeModules;
        if (NativeSpeech && typeof NativeSpeech.stopListening === 'function') {
          const res = await NativeSpeech.stopListening();
          if (res?.text && res.text.trim()) {
            setInput(prev => (prev ? `${prev} ${res.text.trim()}` : res.text.trim()));
          } else {
            Alert.alert(
              'Voice Note',
              'No speech detected. Please speak closer to the microphone.\n\nTip: You can also tap the microphone icon on your iOS keyboard for instant offline dictation anytime!',
            );
          }
        }
      } catch (err: any) {
        Alert.alert(
          'Voice Notice',
          (err?.message || 'Could not finalize voice recording.') +
            '\n\nTip: You can also tap the microphone key on your iOS keyboard for voice input.',
        );
      }
    } else {
      // Start recording
      try {
        const { NativeSpeech } = NativeModules;
        if (NativeSpeech && typeof NativeSpeech.startListening === 'function') {
          const perm = await NativeSpeech.requestPermissions();
          if (!perm?.speech || !perm?.microphone) {
            Alert.alert(
              'Microphone Access',
              'Please allow Speech Recognition and Microphone access in iOS Settings to transcribe offline.',
            );
            return;
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
            'NativeSpeech module is ready in Xcode! Once you rebuild your iOS app, this button transcribes speech offline.\n\nTip: You can also tap the microphone icon on your iOS keyboard for offline dictation right now!',
          );
        }
      } catch (err: any) {
        Alert.alert(
          'Voice Notice',
          (err?.message || 'Could not start voice recording.') +
            '\n\nTip: You can also tap the microphone icon on your iOS keyboard for offline dictation anytime!',
        );
      }
    }
  };

  useEffect(() => {
    prepareModel();
    return () => {
      ctxRef.current?.release();
      ctxRef.current = null;
    };
  }, [prepareModel]);

  // Helper to extract text from chosen image with feedback
  const processImageAttachment = async (asset: { uri?: string; fileName?: string; fileSize?: number }) => {
    if (!asset.uri) return;
    const tempId = `${Date.now()}`;
    const name = asset.fileName || 'photo.jpg';

    // Set initial attachment with extracting state
    setAttachment({
      id: tempId,
      type: 'image',
      uri: asset.uri,
      name,
      size: asset.fileSize,
      isExtracting: true,
    });

    try {
      const text = await extractTextFromImage(asset.uri);
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
    setAttachMenuOpen(false);
    try {
      const res = await launchCamera({
        mediaType: 'photo',
        quality: 0.8,
        saveToPhotos: false,
      });
      if (res.didCancel || !res.assets || !res.assets[0]?.uri) return;
      await processImageAttachment(res.assets[0]);
    } catch (e: any) {
      Alert.alert('Camera Notice', e?.message || 'Could not open camera.');
    }
  };

  const handleChoosePhoto = async () => {
    setAttachMenuOpen(false);
    try {
      const res = await launchImageLibrary({
        mediaType: 'photo',
        quality: 0.8,
        selectionLimit: 1,
      });
      if (res.didCancel || !res.assets || !res.assets[0]?.uri) return;
      await processImageAttachment(res.assets[0]);
    } catch (e: any) {
      Alert.alert('Photo Notice', e?.message || 'Could not select photo.');
    }
  };

  const handlePickDocument = async () => {
    setAttachMenuOpen(false);
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

      const tempId = `${Date.now()}`;
      setAttachment({
        id: tempId,
        type: isImg ? 'image' : 'document',
        uri: doc.uri,
        name: doc.name || (isImg ? 'photo.jpg' : 'document.pdf'),
        size: doc.size ?? undefined,
        isExtracting: true,
      });

      let extracted = '';
      try {
        if (isImg) {
          extracted = await extractTextFromImage(doc.uri);
        } else if (
          lower.endsWith('.txt') ||
          lower.endsWith('.md') ||
          lower.endsWith('.json') ||
          lower.endsWith('.csv')
        ) {
          extracted = await RNFS.readFile(doc.uri, 'utf8');
        } else if (lower.endsWith('.pdf')) {
          const raw = await RNFS.readFile(doc.uri, 'ascii');
          const matches =
            raw.match(/\((.*?)\)\s*Tj/g) || raw.match(/\[(.*?)\]\s*TJ/g);
          if (matches) {
            extracted = matches
              .map(m => m.replace(/[\(\)\[\]]|Tj|TJ/g, '').trim())
              .filter(Boolean)
              .join(' ');
          }
        }
      } catch {}

      setAttachment(prev => {
        if (!prev || prev.id !== tempId) return prev;
        return {
          ...prev,
          extractedText: extracted || undefined,
          isExtracting: false,
        };
      });
    } catch (e: any) {
      if (!isErrorWithCode(e) || e.code !== errorCodes.OPERATION_CANCELED) {
        Alert.alert('File Notice', e?.message || 'Could not select document.');
      }
    }
  };

  // Reliable auto-scroll helper
  const scrollToBottom = useCallback((animated = true) => {
    setTimeout(() => {
      listRef.current?.scrollToEnd({ animated });
    }, 60);
  }, []);

  // Send message
  const send = useCallback(async () => {
    const rawText = input.trim();
    const currentAttach = attachment;
    const ctx = ctxRef.current;
    if ((!rawText && !currentAttach) || !ctx || generating) {
      return;
    }

    // Dismiss keyboard immediately to reveal full chat area
    Keyboard.dismiss();

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

    let streamed = '';
    try {
      const result = await ctx.completion(
        {
          messages: [
            { role: 'system', content: systemPrompt },
            ...history.map((m, idx) => {
              if (idx === history.length - 1 && currentAttach) {
                return { role: m.role, content: promptWithAttachment };
              }
              return { role: m.role, content: m.content };
            }),
          ],
          n_predict: 512,
          temperature: 0.7,
          top_p: 0.9,
          stop: STOP_WORDS,
        },
        data => {
          streamed += data.token;
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
      scrollToBottom(true);
    } catch (e: any) {
      const errorList: Message[] = [
        ...messages,
        userMsg,
        { id: botId, role: 'assistant', content: `⚠️ ${e?.message ?? String(e)}` },
      ];
      setMessages(errorList);
      persistSessionMessages(errorList);
      scrollToBottom(true);
    } finally {
      setGenerating(false);
    }
  }, [input, attachment, messages, generating, systemPrompt, scrollToBottom, persistSessionMessages]);

  const stop = useCallback(() => {
    ctxRef.current?.stopCompletion();
  }, []);

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
  if (phase !== 'ready') {
    return (
      <View style={styles.center}>
        <Text style={styles.logo}>
          <Text style={{ color: C.accent }}>Ilyas</Text>
          <Text style={{ color: '#7C3AED' }}>Ka</Text>
          <Text style={{ color: '#EC4899' }}>Gpt</Text>
        </Text>
        <Text style={styles.subtitle}>{activeModel.name} · On-Device</Text>

        {phase === 'checking' && <Status text="Checking model…" />}

        {phase === 'downloading' && (
          <View style={styles.card}>
            <Text style={styles.cardTitle}>Downloading {activeModel.name}</Text>
            <View style={styles.barTrack}>
              <View
                style={[styles.barFill, { width: `${Math.round(progress * 100)}%` }]}
              />
            </View>
            <Text style={styles.cardText}>
              {Math.round(progress * 100)}% · {formatSize(downloaded)} /{' '}
              {formatSize(activeModel.size)}
            </Text>
            <Text style={styles.hint}>
              Keep the app open and Wi-Fi on. Once downloaded, it works 100% offline forever.
            </Text>
            <ActivityIndicator color={C.accent} style={{ marginTop: 14 }} />

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
            accessibilityLabel="Open Chat History">
            <Menu size={18} color={C.text} />
          </Pressable>

          <Text style={styles.headerTitle} numberOfLines={1}>
            <Text style={{ color: C.accent }}>Ilyas</Text>
            <Text style={{ color: '#7C3AED' }}>Ka</Text>
            <Text style={{ color: '#EC4899' }}>Gpt</Text>
          </Text>

          <Pressable
            style={({ pressed }) => [styles.badge, pressed && styles.pressed]}
            onPress={() => {
              setSettingsTab('models');
              setSettingsOpen(true);
            }}
            accessibilityLabel="Switch AI Model">
            <View style={styles.dot} />
            <Text style={styles.badgeText}>
              {activeModel.id === 'qwen2.5-14b'
                ? '14B Ultra Genius'
                : activeModel.id === 'qwen2.5-7b'
                ? '7B Super Genius'
                : activeModel.id === 'qwen2.5-3b'
                ? '3B Smart AI'
                : '1.5B Fast AI'}
            </Text>
          </Pressable>
        </View>

        <View style={styles.headerRight}>
          <Pressable
            style={({ pressed }) => [
              styles.headerIconBtn,
              pressed && styles.pressed,
            ]}
            onPress={startNewChat}
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
            accessibilityLabel="Settings">
            <Settings size={17} color={C.text} />
          </Pressable>
        </View>
      </View>

      {/* Chat Messages */}
      <FlatList
        ref={listRef}
        data={messages}
        keyExtractor={m => m.id}
        style={{ flex: 1 }}
        contentContainerStyle={styles.list}
        keyboardDismissMode="on-drag"
        keyboardShouldPersistTaps="handled"
        onContentSizeChange={() => scrollToBottom(true)}
        ListEmptyComponent={
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
                  <View style={styles.botBadge}>
                    <Sparkles size={10} color={C.accent} />
                  </View>
                  <Text style={styles.botName}>
                    <Text style={{ color: C.accent }}>Ilyas</Text>
                    <Text style={{ color: '#7C3AED' }}>KaGpt</Text>
                  </Text>
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
                  ? 'Extracting text with OCR…'
                  : attachment.type === 'image'
                  ? attachment.extractedText
                    ? `Text extracted: "${attachment.extractedText.slice(0, 25)}…"`
                    : 'Photo attached · No text found'
                  : 'Document attached' + (attachment.extractedText ? ' · Text extracted' : '')}
              </Text>
            </View>
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
              onPress={send}
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
                ) : (
                  <Settings size={18} color={C.accent} />
                )}
                <Text style={styles.modalTitle}>
                  {settingsTab === 'models' ? 'On-Device AI Models' : 'Personalization & Persona'}
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

            {/* Segmented Tab Bar */}
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
                  size={15}
                  color={settingsTab === 'models' ? C.accent : C.textSecondary}
                />
                <Text
                  style={[
                    styles.tabButtonText,
                    settingsTab === 'models' && styles.tabButtonTextActive,
                  ]}>
                  Model Center
                </Text>
              </Pressable>

              <Pressable
                style={[
                  styles.tabButton,
                  settingsTab === 'persona' && styles.tabButtonActive,
                ]}
                onPress={() => setSettingsTab('persona')}>
                <Settings
                  size={15}
                  color={settingsTab === 'persona' ? C.accent : C.textSecondary}
                />
                <Text
                  style={[
                    styles.tabButtonText,
                    settingsTab === 'persona' && styles.tabButtonTextActive,
                  ]}>
                  Persona & Prompt
                </Text>
              </Pressable>
            </View>

            <ScrollView
              showsVerticalScrollIndicator={false}
              keyboardDismissMode="on-drag"
              keyboardShouldPersistTaps="handled">
              {settingsTab === 'models' ? (
                /* Tab 1: On-Device AI Model Hub (9GB 14B, 4.7GB 7B, 2.1GB 3B, 1.1GB 1.5B) */
                <View>
                  <View style={styles.modelHubHeader}>
                    <Text style={styles.modelSectionTitle}>Available Offline Brains</Text>
                    <Text style={styles.settingHint}>
                      Switch or download models. You are currently running {activeModel.name}.
                    </Text>
                  </View>

                  <View style={{ gap: 10, marginBottom: 16 }}>
                    {AVAILABLE_MODELS.map(m => {
                      const isSelected = selectedModelId === m.id;
                      const isDownloaded = downloadedModels[m.id] === true;
                      const isActive = isSelected && phase === 'ready';

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
                            {isActive ? (
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
                              {isDownloaded && (
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
                              )}

                              {!isDownloaded && (
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
    paddingHorizontal: 20,
    paddingTop: Platform.OS === 'ios' ? 52 : 16,
    paddingBottom: 14,
    borderBottomWidth: 1,
    borderBottomColor: C.border,
    backgroundColor: C.surface,
  },
  headerLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
    flex: 1,
    marginRight: 10,
  },
  headerTitle: {
    color: C.text,
    fontSize: 20,
    fontWeight: '800',
    letterSpacing: -0.3,
    flexShrink: 1,
  },
  badge: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    backgroundColor: C.accentGreenBg,
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 999,
  },
  dot: { width: 7, height: 7, borderRadius: 4, backgroundColor: C.accentGreen },
    badgeText: { color: C.accentGreen, fontSize: 12, fontWeight: '700' },
  headerRight: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
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
    gap: 6,
    marginBottom: 6,
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
});

export default App;
