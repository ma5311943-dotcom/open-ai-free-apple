/**
 * OfflineAI - on-device chat using llama.rn (llama.cpp).
 * Downloads a GGUF model once, then runs 100% offline.
 *
 * @format
 */

import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  ActivityIndicator,
  FlatList,
  KeyboardAvoidingView,
  Platform,
  Pressable,
  SafeAreaView,
  StatusBar,
  StyleSheet,
  Text,
  TextInput,
  View,
} from 'react-native';
import RNFS from 'react-native-fs';
import { initLlama, LlamaContext } from 'llama.rn';

// ---- Model config ---------------------------------------------------------
const MODEL_NAME = 'Qwen2.5 1.5B Instruct';
const MODEL_FILE = 'qwen2.5-1.5b-instruct-q4_k_m.gguf';
const MODEL_URL =
  'https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/' +
  MODEL_FILE;
const MODEL_SIZE = 1117320736; // bytes, used to detect incomplete downloads
const MODEL_PATH = `${RNFS.DocumentDirectoryPath}/${MODEL_FILE}`;
const TEMP_PATH = `${MODEL_PATH}.part`;

const SYSTEM_PROMPT =
  'You are a helpful, concise assistant running fully offline on a phone. ' +
  'Answer clearly. Use short code examples when useful.';
const STOP_WORDS = ['<|im_end|>', '<|endoftext|>', '</s>', '<|eot_id|>'];
const MAX_HISTORY = 8; // last N messages sent to the model (keeps context small)

type Phase = 'checking' | 'downloading' | 'loading' | 'ready' | 'error';
type Role = 'user' | 'assistant';
type Message = { id: string; role: Role; content: string };

const C = {
  bg: '#0B0D14',
  surface: '#151926',
  surface2: '#1D2233',
  border: '#262C40',
  text: '#E8EAF2',
  muted: '#8A90A6',
  accent: '#7C6CF6',
  accent2: '#22D3EE',
  danger: '#F87171',
};

function formatMB(bytes: number) {
  return `${(bytes / 1024 / 1024).toFixed(0)} MB`;
}

function App() {
  return (
    <SafeAreaView style={styles.root}>
      <StatusBar barStyle="light-content" />
      <ChatScreen />
    </SafeAreaView>
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

  const ctxRef = useRef<LlamaContext | null>(null);
  const listRef = useRef<FlatList<Message>>(null);

  // ---- Download (once) + load model --------------------------------------
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

  // ---- Chat ---------------------------------------------------------------
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
            { role: 'system', content: SYSTEM_PROMPT },
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
  }, [input, messages, generating]);

  const stop = useCallback(() => {
    ctxRef.current?.stopCompletion();
  }, []);

  // ---- UI -----------------------------------------------------------------
  if (phase !== 'ready') {
    return (
      <View style={styles.center}>
        <Text style={styles.logo}>◆ OfflineAI</Text>
        <Text style={styles.subtitle}>{MODEL_NAME} · on-device</Text>

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
      <View style={styles.header}>
        <Text style={styles.headerTitle}>◆ OfflineAI</Text>
        <View style={styles.badge}>
          <View style={styles.dot} />
          <Text style={styles.badgeText}>Offline ready</Text>
        </View>
      </View>

      <FlatList
        ref={listRef}
        data={messages}
        keyExtractor={m => m.id}
        contentContainerStyle={styles.list}
        onContentSizeChange={() => listRef.current?.scrollToEnd({ animated: true })}
        ListEmptyComponent={
          <View style={styles.empty}>
            <Text style={styles.emptyTitle}>Ask anything</Text>
            <Text style={styles.hint}>
              Try: "What is a constructor in Java?"
            </Text>
          </View>
        }
        renderItem={({ item }) => (
          <View
            style={[
              styles.bubble,
              item.role === 'user' ? styles.userBubble : styles.botBubble,
            ]}>
            <Text style={styles.bubbleText} selectable>
              {item.content || '…'}
            </Text>
          </View>
        )}
      />

      {!!speed && <Text style={styles.speed}>{speed}</Text>}

      <View style={styles.inputRow}>
        <TextInput
          style={styles.input}
          value={input}
          onChangeText={setInput}
          placeholder="Message…"
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
            <Text style={styles.sendText}>■</Text>
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
            <Text style={styles.sendText}>➤</Text>
          </Pressable>
        )}
      </View>
    </KeyboardAvoidingView>
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

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: C.bg },
  center: {
    flex: 1,
    backgroundColor: C.bg,
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 24,
  },
  logo: { color: C.text, fontSize: 32, fontWeight: '800', letterSpacing: 0.5 },
  subtitle: { color: C.muted, marginTop: 6, marginBottom: 28 },
  status: { flexDirection: 'row', alignItems: 'center', gap: 10 },
  card: {
    width: '100%',
    backgroundColor: C.surface,
    borderColor: C.border,
    borderWidth: 1,
    borderRadius: 18,
    padding: 20,
    gap: 12,
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
  pressed: { opacity: 0.7, transform: [{ scale: 0.97 }] },
  disabled: { opacity: 0.4 },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 18,
    paddingTop: Platform.OS === 'ios' ? 14 : 12,
    paddingBottom: 14,
    borderBottomWidth: 1,
    borderBottomColor: C.border,
    backgroundColor: C.surface,
  },
  headerTitle: { color: C.text, fontSize: 20, fontWeight: '800' },
  badge: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    backgroundColor: C.surface2,
    paddingHorizontal: 10,
    paddingVertical: 5,
    borderRadius: 999,
  },
  dot: { width: 7, height: 7, borderRadius: 4, backgroundColor: C.accent2 },
  badgeText: { color: C.muted, fontSize: 12 },
  list: {
    paddingHorizontal: 16,
    paddingTop: 16,
    paddingBottom: 16,
    gap: 12,
    flexGrow: 1,
  },
  empty: { flex: 1, alignItems: 'center', justifyContent: 'center', gap: 8 },
  emptyTitle: { color: C.text, fontSize: 22, fontWeight: '700' },
  bubble: {
    maxWidth: '86%',
    paddingHorizontal: 16,
    paddingVertical: 12,
    borderRadius: 18,
    marginVertical: 4,
  },
  userBubble: {
    alignSelf: 'flex-end',
    backgroundColor: C.accent,
    borderBottomRightRadius: 4,
  },
  botBubble: {
    alignSelf: 'flex-start',
    backgroundColor: C.surface2,
    borderColor: C.border,
    borderWidth: 1,
    borderBottomLeftRadius: 4,
  },
  bubbleText: { color: C.text, fontSize: 15, lineHeight: 22 },
  speed: { color: C.muted, fontSize: 11, textAlign: 'center', marginBottom: 4 },
  inputRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
    paddingHorizontal: 16,
    paddingTop: 12,
    paddingBottom: Platform.OS === 'ios' ? 28 : 14,
    borderTopWidth: 1,
    borderTopColor: C.border,
    backgroundColor: C.surface,
  },
  input: {
    flex: 1,
    minHeight: 44,
    maxHeight: 120,
    color: C.text,
    backgroundColor: C.surface2,
    borderRadius: 22,
    paddingHorizontal: 16,
    paddingTop: 10,
    paddingBottom: 10,
    fontSize: 15,
  },
  sendBtn: {
    width: 44,
    height: 44,
    borderRadius: 22,
    backgroundColor: C.accent,
    alignItems: 'center',
    justifyContent: 'center',
  },
  stopBtn: { backgroundColor: C.danger },
  sendText: { color: '#fff', fontSize: 18, fontWeight: '700' },
});

export default App;
