package com.offlineai

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import androidx.core.content.ContextCompat
import com.facebook.react.bridge.Arguments
import com.facebook.react.bridge.Promise
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.bridge.ReactContextBaseJavaModule
import com.facebook.react.bridge.ReactMethod
import com.facebook.react.bridge.ReadableMap
import com.facebook.react.bridge.WritableMap
import com.facebook.react.modules.core.DeviceEventManagerModule
import java.util.Locale

class NativeSpeechModule(private val reactContext: ReactApplicationContext) :
    ReactContextBaseJavaModule(reactContext),
    RecognitionListener,
    TextToSpeech.OnInitListener {

    private val mainHandler = Handler(Looper.getMainLooper())

    // Speech-to-Text state
    private var speechRecognizer: SpeechRecognizer? = null
    private var currentTranscription: String = ""
    private var stopPromise: Promise? = null
    private var fallbackStopRunnable: Runnable? = null

    // Text-to-Speech state
    private var tts: TextToSpeech? = null
    private var isTtsReady = false
    private var pendingSpeakText: String? = null
    private var pendingSpeakPromise: Promise? = null

    init {
        mainHandler.post {
            try {
                tts = TextToSpeech(reactContext.applicationContext, this)
            } catch (e: Exception) {
                // Ignore initialization error
            }
        }
    }

    override fun getName(): String = "NativeSpeech"

    private fun sendEvent(eventName: String, params: WritableMap?) {
        try {
            reactContext
                .getJSModule(DeviceEventManagerModule.RCTDeviceEventEmitter::class.java)
                .emit(eventName, params)
        } catch (e: Exception) {
            // Context might not be ready
        }
    }

    // =========================================================================
    // SPEECH RECOGNITION (STT) METHODS
    // =========================================================================

    @ReactMethod
    fun requestPermissions(promise: Promise) {
        val micGranted = ContextCompat.checkSelfPermission(
            reactContext,
            Manifest.permission.RECORD_AUDIO
        ) == PackageManager.PERMISSION_GRANTED

        val result = Arguments.createMap()
        result.putBoolean("speech", true)
        result.putBoolean("microphone", micGranted)
        promise.resolve(result)
    }

    @ReactMethod
    fun isAvailable(promise: Promise) {
        mainHandler.post {
            val available = SpeechRecognizer.isRecognitionAvailable(reactContext)
            val result = Arguments.createMap()
            result.putBoolean("available", available)
            result.putBoolean("onDevice", true)
            result.putString("locale", Locale.getDefault().toLanguageTag())
            promise.resolve(result)
        }
    }

    @ReactMethod
    fun startListening(localeIdentifier: String, promise: Promise) {
        val micGranted = ContextCompat.checkSelfPermission(
            reactContext,
            Manifest.permission.RECORD_AUDIO
        ) == PackageManager.PERMISSION_GRANTED

        if (!micGranted) {
            promise.reject("PERMISSION_DENIED", "Microphone permission is not granted.")
            return
        }

        mainHandler.post {
            try {
                cleanupRecognizer()

                val recognizer = SpeechRecognizer.createSpeechRecognizer(reactContext)
                speechRecognizer = recognizer
                recognizer.setRecognitionListener(this)
                currentTranscription = ""

                val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
                    putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
                    putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, true)
                    putExtra(RecognizerIntent.EXTRA_MAX_RESULTS, 1)
                    // Request offline on-device speech model
                    putExtra(RecognizerIntent.EXTRA_PREFER_OFFLINE, true)

                    if (localeIdentifier.isNotBlank()) {
                        putExtra(RecognizerIntent.EXTRA_LANGUAGE, localeIdentifier)
                        putExtra(RecognizerIntent.EXTRA_LANGUAGE_PREFERENCE, localeIdentifier)
                    } else {
                        val currentLocale = Locale.getDefault().toLanguageTag()
                        putExtra(RecognizerIntent.EXTRA_LANGUAGE, currentLocale)
                        putExtra(RecognizerIntent.EXTRA_LANGUAGE_PREFERENCE, currentLocale)
                    }
                }

                recognizer.startListening(intent)

                val result = Arguments.createMap()
                result.putString("status", "recording")
                result.putString("locale", Locale.getDefault().toLanguageTag())
                result.putBoolean("onDevice", true)
                promise.resolve(result)
            } catch (e: Exception) {
                cleanupRecognizer()
                promise.reject("START_FAILED", "Failed to start speech recognition: ${e.message}", e)
            }
        }
    }

    @ReactMethod
    fun stopListening(promise: Promise) {
        mainHandler.post {
            if (speechRecognizer == null) {
                val res = Arguments.createMap()
                res.putString("text", currentTranscription.trim())
                promise.resolve(res)
                return@post
            }

            stopPromise = promise

            try {
                speechRecognizer?.stopListening()
            } catch (e: Exception) {
                // If stopListening fails, resolve current transcription immediately
                val res = Arguments.createMap()
                res.putString("text", currentTranscription.trim())
                promise.resolve(res)
                stopPromise = null
                cleanupRecognizer()
                return@post
            }

            // Fallback: If onResults or onError doesn't fire within 800ms, resolve immediately
            fallbackStopRunnable = Runnable {
                stopPromise?.let { p ->
                    val res = Arguments.createMap()
                    res.putString("text", currentTranscription.trim())
                    p.resolve(res)
                    stopPromise = null
                }
                cleanupRecognizer()
            }
            mainHandler.postDelayed(fallbackStopRunnable!!, 800)
        }
    }

    @ReactMethod
    fun cancelListening(promise: Promise) {
        mainHandler.post {
            cleanupRecognizer()
            currentTranscription = ""
            val res = Arguments.createMap()
            res.putString("status", "cancelled")
            promise.resolve(res)
        }
    }

    private fun cleanupRecognizer() {
        fallbackStopRunnable?.let { mainHandler.removeCallbacks(it) }
        fallbackStopRunnable = null

        speechRecognizer?.let {
            try {
                it.cancel()
                it.destroy()
            } catch (e: Exception) {}
        }
        speechRecognizer = null
    }

    // RecognitionListener callbacks
    override fun onReadyForSpeech(params: Bundle?) {
        sendEvent("onSpeechStart", null)
    }

    override fun onBeginningOfSpeech() {}
    override fun onRmsChanged(rmsdB: Float) {}
    override fun onBufferReceived(buffer: ByteArray?) {}
    override fun onEndOfSpeech() {}

    override fun onError(error: Int) {
        fallbackStopRunnable?.let { mainHandler.removeCallbacks(it) }
        fallbackStopRunnable = null

        stopPromise?.let { p ->
            val res = Arguments.createMap()
            res.putString("text", currentTranscription.trim())
            p.resolve(res)
            stopPromise = null
        }

        val eventData = Arguments.createMap()
        eventData.putInt("error", error)
        sendEvent("onSpeechError", eventData)
        cleanupRecognizer()
    }

    override fun onResults(results: Bundle?) {
        fallbackStopRunnable?.let { mainHandler.removeCallbacks(it) }
        fallbackStopRunnable = null

        val matches = results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
        if (!matches.isNullOrEmpty()) {
            currentTranscription = matches[0]
        }

        stopPromise?.let { p ->
            val res = Arguments.createMap()
            res.putString("text", currentTranscription.trim())
            p.resolve(res)
            stopPromise = null
        }

        val eventData = Arguments.createMap()
        eventData.putString("text", currentTranscription.trim())
        sendEvent("onSpeechResults", eventData)
        cleanupRecognizer()
    }

    override fun onPartialResults(partialResults: Bundle?) {
        val matches = partialResults?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
        if (!matches.isNullOrEmpty()) {
            currentTranscription = matches[0]
            val eventData = Arguments.createMap()
            eventData.putString("text", currentTranscription)
            sendEvent("onSpeechPartialResults", eventData)
        }
    }

    override fun onEvent(eventType: Int, params: Bundle?) {}

    // =========================================================================
    // TEXT-TO-SPEECH (TTS) METHODS
    // =========================================================================

    override fun onInit(status: Int) {
        if (status == TextToSpeech.SUCCESS) {
            isTtsReady = true
            tts?.language = Locale.getDefault()
            tts?.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
                override fun onStart(utteranceId: String?) {
                    val map = Arguments.createMap()
                    map.putString("utteranceId", utteranceId ?: "")
                    sendEvent("onTtsStart", map)
                }

                override fun onDone(utteranceId: String?) {
                    val map = Arguments.createMap()
                    map.putString("utteranceId", utteranceId ?: "")
                    sendEvent("onTtsDone", map)
                }

                override fun onError(utteranceId: String?) {
                    val map = Arguments.createMap()
                    map.putString("utteranceId", utteranceId ?: "")
                    sendEvent("onTtsError", map)
                }
            })

            // If there was a pending speak call during init
            pendingSpeakText?.let { text ->
                val promise = pendingSpeakPromise
                pendingSpeakText = null
                pendingSpeakPromise = null
                speakInternal(text, null, promise)
            }
        }
    }

    @ReactMethod
    fun speak(text: String, options: ReadableMap?, promise: Promise) {
        val cleanText = text.trim()
        if (cleanText.isEmpty()) {
            val res = Arguments.createMap()
            res.putString("status", "empty")
            promise.resolve(res)
            return
        }

        mainHandler.post {
            if (!isTtsReady || tts == null) {
                // Save pending speak
                pendingSpeakText = cleanText
                pendingSpeakPromise = promise
                return@post
            }
            speakInternal(cleanText, options, promise)
        }
    }

    private fun speakInternal(text: String, options: ReadableMap?, promise: Promise?) {
        try {
            val engine = tts ?: run {
                promise?.reject("TTS_NOT_INITIALIZED", "TextToSpeech engine is not ready.")
                return
            }

            // Options: rate, pitch, language
            if (options != null) {
                if (options.hasKey("rate")) {
                    engine.setSpeechRate(options.getDouble("rate").toFloat())
                }
                if (options.hasKey("pitch")) {
                    engine.setPitch(options.getDouble("pitch").toFloat())
                }
                if (options.hasKey("language")) {
                    val lang = options.getString("language")
                    if (!lang.isNullOrBlank()) {
                        val locale = Locale.forLanguageTag(lang)
                        if (engine.isLanguageAvailable(locale) >= TextToSpeech.LANG_AVAILABLE) {
                            engine.language = locale
                        }
                    }
                }
            }

            val utteranceId = "utt_${System.currentTimeMillis()}"
            val params = Bundle()
            params.putString(TextToSpeech.Engine.KEY_PARAM_UTTERANCE_ID, utteranceId)

            val queueMode = if (options != null && options.hasKey("queueMode") && options.getString("queueMode") == "add") {
                TextToSpeech.QUEUE_ADD
            } else {
                TextToSpeech.QUEUE_FLUSH
            }

            engine.speak(text, queueMode, params, utteranceId)

            val result = Arguments.createMap()
            result.putString("status", "speaking")
            result.putString("utteranceId", utteranceId)
            promise?.resolve(result)
        } catch (e: Exception) {
            promise?.reject("TTS_SPEAK_ERROR", e.message, e)
        }
    }

    @ReactMethod
    fun stopSpeaking(promise: Promise) {
        mainHandler.post {
            try {
                tts?.stop()
                val result = Arguments.createMap()
                result.putString("status", "stopped")
                promise.resolve(result)

                val map = Arguments.createMap()
                map.putString("utteranceId", "stopped")
                sendEvent("onTtsDone", map)
            } catch (e: Exception) {
                promise.reject("TTS_STOP_ERROR", e.message, e)
            }
        }
    }

    @ReactMethod
    fun isSpeaking(promise: Promise) {
        val speaking = tts?.isSpeaking == true
        val result = Arguments.createMap()
        result.putBoolean("speaking", speaking)
        promise.resolve(result)
    }

    override fun invalidate() {
        super.invalidate()
        mainHandler.post {
            cleanupRecognizer()
            tts?.stop()
            tts?.shutdown()
            tts = null
            isTtsReady = false
        }
    }
}
