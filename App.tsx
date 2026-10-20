/**
 * OfflineAI - On-device AI assistant using llama.rn (llama.cpp).
 * Option 2: Modern Apple Light Theme with Lucide React Native Vector Icons.
 *
 * @format
 */

import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  ActivityIndicator,
  FlatList,
  KeyboardAvoidingView,
  Modal,
  Platform,
  Pressable,
  SafeAreaView,
  ScrollView,
  Share,
  StatusBar,
  StyleSheet,
  Text,
  TextInput,
  View,
} from 'react-native';
import RNFS from 'react-native-fs';
import { initLlama, LlamaContext } from 'llama.rn';
import {
  Brain,
  Check,
  Code2,
  Copy,
  FileText,
  Lightbulb,
  Mail,
  Send,
  Settings,
  Sparkles,
  Square,
  Trash2,
  X,
  Zap,
} from 'lucide-react-native';

// ---- Model configuration --------------------------------------------------
const MODEL_NAME = 'Qwen2.5 1.5B Instruct';
const MODEL_FILE = 'qwen2.5-1.5b-instruct-q4_k_m.gguf';
const MODEL_URL =
  'https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/' +
  MODEL_FILE;
const MODEL_SIZE = 1117320736;
const MODEL_PATH = `${RNFS.DocumentDirectoryPath}/${MODEL_FILE}`;
const TEMP_PATH = `${MODEL_PATH}.part`;
const SETTINGS_PATH = `${RNFS.DocumentDirectoryPath}/settings.json`;

const SYSTEM_PROMPT =
  'You are a helpful, concise assistant running fully offline on a phone. ' +
  'Answer clearly. Use short code examples when useful.';
const STOP_WORDS = ['<|im_end|>', '<|endoftext|>', '</s>', '<|eot_id|>'];
const MAX_HISTORY = 8;

type Phase = 'checking' | 'downloading' | 'loading' | 'ready' | 'error';
type Role = 'user' | 'assistant';
type Message = { id: string; role: Role; content: string };

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

function formatMB(bytes: number) {
  return `${(bytes / 1024 / 1024).toFixed(0)} MB`;
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
  const [tempPrompt, setTempPrompt] = useState(SYSTEM_PROMPT);

  const ctxRef = useRef<LlamaContext | null>(null);
  const listRef = useRef<FlatList<Message>>(null);

  // Load saved personalization
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
          })
          .catch(() => {});
      }
    });
  }, []);

  const saveSettings = useCallback(async () => {
    const nextPrompt = tempPrompt.trim() || SYSTEM_PROMPT;
    setSystemPrompt(nextPrompt);
    setSettingsOpen(false);
    try {
      await RNFS.writeFile(
        SETTINGS_PATH,
        JSON.stringify({ systemPrompt: nextPrompt }),
        'utf8',
      );
    } catch {}
  }, [tempPrompt]);

  // Model preparation
  const prepareModel = useCallback(async () => {
    try {
      setError('');
      setPhase('checking');

      let ready = false;
      if (await RNFS.exists(MODEL_PATH)) {
        const stat = await RNFS.stat(MODEL_PATH);
        ready = Number(stat.size) === MODEL_SIZE;
        if (!ready) {
          await RNFS.unlink(MODEL_PATH);
        }
      }

      if (!ready) {
        setPhase('downloading');
        if (await RNFS.exists(TEMP_PATH)) {
          await RNFS.unlink(TEMP_PATH);
        }
        const job = RNFS.downloadFile({
          fromUrl: MODEL_URL,
          toFile: TEMP_PATH,
          begin: () => {},
          progressInterval: 250,
          progress: res => {
            const total =
              res.contentLength && res.contentLength > 100000000
                ? res.contentLength
                : MODEL_SIZE;
            setDownloaded(res.bytesWritten);
            setProgress(Math.min(1, Math.max(0, res.bytesWritten / total)));
          },
        });
        const result = await job.promise;
        if (result.statusCode !== 200) {
          throw new Error(`Download failed (HTTP ${result.statusCode})`);
        }
        const stat = await RNFS.stat(TEMP_PATH);
        if (Number(stat.size) !== MODEL_SIZE) {
          throw new Error('Download incomplete. Check internet and retry.');
        }
        await RNFS.moveFile(TEMP_PATH, MODEL_PATH);
      }

      setPhase('loading');
      const ctx = await initLlama({
        model: `file://${MODEL_PATH}`,
        n_ctx: 2048,
        n_gpu_layers: 0,
        use_mlock: false,
      });
      ctxRef.current = ctx;
      setPhase('ready');
    } catch (e: any) {
      setError(e?.message ?? String(e));
      setPhase('error');
    }
  }, []);

  useEffect(() => {
    prepareModel();
    return () => {
      ctxRef.current?.release();
      ctxRef.current = null;
    };
  }, [prepareModel]);

  // Send message
  const send = useCallback(async () => {
    const text = input.trim();
    const ctx = ctxRef.current;
    if (!text || !ctx || generating) {
      return;
    }

    const userMsg: Message = { id: `${Date.now()}u`, role: 'user', content: text };
    const botId = `${Date.now()}a`;
    const history = [...messages, userMsg].slice(-MAX_HISTORY);

    setInput('');
    setSpeed('');
    setGenerating(true);
    setMessages(prev => [
      ...prev,
      userMsg,
      { id: botId, role: 'assistant', content: '' },
    ]);

    let streamed = '';
    try {
      const result = await ctx.completion(
        {
          messages: [
            { role: 'system', content: systemPrompt },
            ...history.map(m => ({ role: m.role, content: m.content })),
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
      setMessages(prev =>
        prev.map(m => (m.id === botId ? { ...m, content: finalText } : m)),
      );
      const tps = result.timings?.predicted_per_second;
      if (tps) {
        setSpeed(`${tps.toFixed(1)} tokens/s`);
      }
    } catch (e: any) {
      setMessages(prev =>
        prev.map(m =>
          m.id === botId
            ? { ...m, content: `⚠️ ${e?.message ?? String(e)}` }
            : m,
        ),
      );
    } finally {
      setGenerating(false);
    }
  }, [input, messages, generating, systemPrompt]);

  const stop = useCallback(() => {
    ctxRef.current?.stopCompletion();
  }, []);

  // ---- UI -----------------------------------------------------------------
  if (phase !== 'ready') {
    return (
      <View style={styles.center}>
        <Text style={styles.logo}>OfflineAI</Text>
        <Text style={styles.subtitle}>{MODEL_NAME} · On-Device</Text>

        {phase === 'checking' && <Status text="Checking model…" />}

        {phase === 'downloading' && (
          <View style={styles.card}>
            <Text style={styles.cardTitle}>Downloading model (one time)</Text>
            <View style={styles.barTrack}>
              <View
                style={[styles.barFill, { width: `${Math.round(progress * 100)}%` }]}
              />
            </View>
            <Text style={styles.cardText}>
              {Math.round(progress * 100)}% · {formatMB(downloaded)} /{' '}
              {formatMB(MODEL_SIZE)}
            </Text>
            <Text style={styles.hint}>
              Keep the app open and Wi-Fi on. After this, it works offline.
            </Text>
            <ActivityIndicator color={C.accent} style={{ marginTop: 14 }} />
          </View>
        )}

        {phase === 'loading' && <Status text="Loading model into memory…" />}

        {phase === 'error' && (
          <View style={styles.card}>
            <Text style={[styles.cardTitle, { color: C.danger }]}>Error</Text>
            <Text style={styles.cardText}>{error}</Text>
            <Pressable
              style={({ pressed }) => [styles.button, pressed && styles.pressed]}
              onPress={prepareModel}>
              <Text style={styles.buttonText}>Retry</Text>
            </Pressable>
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
          <Text style={styles.headerTitle}>OfflineAI</Text>
          <View style={styles.badge}>
            <View style={styles.dot} />
            <Text style={styles.badgeText}>On-Device AI</Text>
          </View>
        </View>
        <Pressable
          style={({ pressed }) => [styles.settingsBtn, pressed && styles.pressed]}
          onPress={() => {
            setTempPrompt(systemPrompt);
            setSettingsOpen(true);
          }}>
          <Settings size={18} color={C.text} />
        </Pressable>
      </View>

      {/* Chat Messages */}
      <FlatList
        ref={listRef}
        data={messages}
        keyExtractor={m => m.id}
        contentContainerStyle={styles.list}
        onContentSizeChange={() => listRef.current?.scrollToEnd({ animated: true })}
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
        renderItem={({ item }) => (
          <View
            style={[
              styles.bubble,
              item.role === 'user' ? styles.userBubble : styles.botBubble,
            ]}>
            {item.role === 'assistant' && (
              <View style={styles.botHeader}>
                <View style={styles.botBadge}>
                  <Sparkles size={10} color={C.accent} />
                </View>
                <Text style={styles.botName}>OfflineAI</Text>
              </View>
            )}
            <MessageContent content={item.content} role={item.role} />
          </View>
        )}
      />

      {!!speed && <Text style={styles.speed}>{speed}</Text>}

      {/* Floating Capsule Input Bar */}
      <View style={styles.inputContainer}>
        <View style={styles.inputCard}>
          <TextInput
            style={styles.input}
            value={input}
            onChangeText={setInput}
            placeholder="Ask OfflineAI…"
            placeholderTextColor={C.muted}
            multiline
            editable={!generating}
          />
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
                !input.trim() && styles.disabled,
                pressed && styles.pressed,
              ]}
              onPress={send}
              disabled={!input.trim()}>
              <Send size={15} color="#FFFFFF" style={{ marginLeft: 2 }} />
            </Pressable>
          )}
        </View>
      </View>

      {/* Settings Modal (Light Theme) */}
      <Modal
        visible={settingsOpen}
        animationType="slide"
        transparent
        onRequestClose={() => setSettingsOpen(false)}>
        <SafeAreaView style={styles.modalOverlay}>
          <View style={styles.modalContent}>
            <View style={styles.modalHeader}>
              <View style={styles.modalTitleRow}>
                <Settings size={18} color={C.text} />
                <Text style={styles.modalTitle}>AI Personalization</Text>
              </View>
              <Pressable onPress={() => setSettingsOpen(false)}>
                <X size={20} color={C.muted} />
              </Pressable>
            </View>

            <ScrollView showsVerticalScrollIndicator={false}>
              <Text style={styles.settingLabel}>Custom Instructions & Persona</Text>
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
                  <Text style={styles.saveBtnText}>Save</Text>
                </Pressable>
              </View>
            </ScrollView>
          </View>
        </SafeAreaView>
      </Modal>
    </KeyboardAvoidingView>
  );
}

// ---- Markdown Code Block & Message Renderer -------------------------------
function MessageContent({ content, role }: { content: string; role: Role }) {
  if (role === 'user' || !content.includes('```')) {
    return (
      <Text
        style={role === 'user' ? styles.userText : styles.botText}
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
      <ScrollView horizontal showsHorizontalScrollIndicator={false}>
        <Text style={styles.codeContent} selectable>
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
    gap: 12,
  },
  headerTitle: { color: C.text, fontSize: 22, fontWeight: '800', letterSpacing: -0.3 },
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
  settingsBtn: {
    width: 38,
    height: 38,
    borderRadius: 19,
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

  // Code Block in Dark Obsidian Theme with Copy
  codeCard: {
    backgroundColor: C.codeBg,
    borderRadius: 14,
    borderWidth: 1,
    borderColor: C.codeBorder,
    overflow: 'hidden',
    marginTop: 8,
    marginBottom: 4,
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

  // Settings Modal
  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.5)',
    justifyContent: 'flex-end',
  },
  modalContent: {
    backgroundColor: C.surface,
    borderTopLeftRadius: 28,
    borderTopRightRadius: 28,
    padding: 22,
    maxHeight: '85%',
    shadowColor: '#000',
    shadowOpacity: 0.1,
    shadowOffset: { width: 0, height: -4 },
    shadowRadius: 16,
    elevation: 10,
  },
  modalHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingBottom: 14,
    borderBottomWidth: 1,
    borderBottomColor: C.border,
    marginBottom: 16,
  },
  modalTitleRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  modalTitle: { color: C.text, fontSize: 18, fontWeight: '800' },
  settingLabel: { color: C.text, fontSize: 15, fontWeight: '700', marginBottom: 4 },
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
});

export default App;
