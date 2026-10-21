import Foundation
import AVFoundation
import Speech

@objc(NativeSpeech)
class NativeSpeech: NSObject, SFSpeechRecognizerDelegate, AVSpeechSynthesizerDelegate {

  private var audioEngine = AVAudioEngine()
  private var speechRecognizer: SFSpeechRecognizer?
  private var recognitionRequest: SFSpeechAudioBufferRecognitionRequest?
  private var recognitionTask: SFSpeechRecognitionTask?
  private var currentTranscription: String = ""
  private var isTapInstalled = false
  private var speechSynthesizer = AVSpeechSynthesizer()

  @objc
  static func requiresMainQueueSetup() -> Bool {
    return true
  }

  // Request Speech and Microphone permissions
  @objc(requestPermissions:rejecter:)
  func requestPermissions(
    _ resolve: @escaping RCTPromiseResolveBlock,
    rejecter reject: @escaping RCTPromiseRejectBlock
  ) {
    SFSpeechRecognizer.requestAuthorization { authStatus in
      DispatchQueue.main.async {
        switch authStatus {
        case .authorized:
          AVAudioSession.sharedInstance().requestRecordPermission { granted in
            DispatchQueue.main.async {
              resolve(["speech": true, "microphone": granted])
            }
          }
        case .denied, .restricted, .notDetermined:
          resolve(["speech": false, "microphone": false])
        @unknown default:
          resolve(["speech": false, "microphone": false])
        }
      }
    }
  }

  // Intelligently find the best supported SFSpeechRecognizer for the user's language/locale
  // (Prevents failures on non-standard Apple locales like en_PK by falling back to en-US/en-GB)
  private func resolveSpeechRecognizer(preferredLocale: String?) -> SFSpeechRecognizer? {
    let supported = SFSpeechRecognizer.supportedLocales()

    // 1. If explicit locale passed
    if let pref = preferredLocale, !pref.isEmpty {
      let loc = Locale(identifier: pref)
      if let recognizer = SFSpeechRecognizer(locale: loc), recognizer.isAvailable {
        return recognizer
      }
      if let recognizer = SFSpeechRecognizer(locale: loc) {
        return recognizer
      }
    }

    // 2. Try current device locale if supported
    let current = Locale.current
    let currentId = current.identifier.replacingOccurrences(of: "_", with: "-")
    if supported.contains(where: { $0.identifier.replacingOccurrences(of: "_", with: "-") == currentId }) {
      if let recognizer = SFSpeechRecognizer(locale: current), recognizer.isAvailable {
        return recognizer
      }
    }

    // 3. Fallback based on language code (e.g. 'en' for 'en_PK' or 'en-PK')
    let langCode = current.languageCode ?? (current as NSLocale).object(forKey: .languageCode) as? String ?? "en"

    if langCode.lowercased() == "en" {
      let commonEnglish = ["en-US", "en_US", "en-GB", "en_GB", "en-IN", "en_IN", "en-AU", "en-CA"]
      for id in commonEnglish {
        let loc = Locale(identifier: id)
        if let recognizer = SFSpeechRecognizer(locale: loc), recognizer.isAvailable {
          return recognizer
        }
      }
      for id in commonEnglish {
        let loc = Locale(identifier: id)
        if let recognizer = SFSpeechRecognizer(locale: loc) {
          return recognizer
        }
      }
    }

    // Match any supported locale sharing the same language code
    for loc in supported {
      let locLang = loc.languageCode ?? (loc as NSLocale).object(forKey: .languageCode) as? String ?? ""
      if locLang.lowercased() == langCode.lowercased() {
        if let recognizer = SFSpeechRecognizer(locale: loc), recognizer.isAvailable {
          return recognizer
        }
      }
    }

    // 4. Default standard fallback: en-US
    let usLocale = Locale(identifier: "en-US")
    if let recognizer = SFSpeechRecognizer(locale: usLocale) {
      return recognizer
    }

    // 5. System default initializer
    if let recognizer = SFSpeechRecognizer() {
      return recognizer
    }

    // 6. First supported locale if any exist
    if let firstSupported = supported.first, let recognizer = SFSpeechRecognizer(locale: firstSupported) {
      return recognizer
    }

    return nil
  }

  // Start listening & transcribing speech (Offline on-device preferred)
  @objc(startListening:resolver:rejecter:)
  func startListening(
    _ localeIdentifier: String,
    resolver resolve: @escaping RCTPromiseResolveBlock,
    rejecter reject: @escaping RCTPromiseRejectBlock
  ) {
    DispatchQueue.main.async { [weak self] in
      guard let self = self else { return }

      // Stop any existing task
      self.cleanupAudio()

      guard let recognizer = self.resolveSpeechRecognizer(preferredLocale: localeIdentifier) else {
        reject("SPEECH_UNAVAILABLE", "Speech recognizer is not available on this device", nil)
        return
      }

      self.speechRecognizer = recognizer
      recognizer.delegate = self
      self.currentTranscription = ""

      do {
        let audioSession = AVAudioSession.sharedInstance()
        try audioSession.setCategory(.playAndRecord, mode: .measurement, options: [.duckOthers, .defaultToSpeaker])
        try audioSession.setActive(true, options: .notifyOthersOnDeactivation)

        let request = SFSpeechAudioBufferRecognitionRequest()
        self.recognitionRequest = request

        // Allow on-device offline recognition when supported without throwing fatal error if assets not yet cached
        if #available(iOS 13.0, *) {
          if recognizer.supportsOnDeviceRecognition {
            request.requiresOnDeviceRecognition = false
          }
        }

        request.shouldReportPartialResults = true

        let inputNode = self.audioEngine.inputNode
        let recordingFormat = inputNode.outputFormat(forBus: 0)

        if self.isTapInstalled {
          inputNode.removeTap(onBus: 0)
          self.isTapInstalled = false
        }

        inputNode.installTap(onBus: 0, bufferSize: 1024, format: recordingFormat) { [weak self] buffer, _ in
          self?.recognitionRequest?.append(buffer)
        }
        self.isTapInstalled = true

        self.audioEngine.prepare()
        try self.audioEngine.start()

        self.recognitionTask = recognizer.recognitionTask(with: request) { [weak self] result, error in
          guard let self = self else { return }
          if let result = result {
            self.currentTranscription = result.bestTranscription.formattedString
          }
          if error != nil || (result?.isFinal ?? false) {
            self.cleanupAudio()
          }
        }

        resolve([
          "status": "recording",
          "locale": recognizer.locale.identifier,
          "onDevice": recognizer.supportsOnDeviceRecognition
        ])
      } catch {
        self.cleanupAudio()
        reject("START_FAILED", "Failed to start audio engine: \(error.localizedDescription)", error)
      }
    }
  }

  // Stop listening and resolve with final transcribed text
  @objc(stopListening:rejecter:)
  func stopListening(
    _ resolve: @escaping RCTPromiseResolveBlock,
    rejecter reject: @escaping RCTPromiseRejectBlock
  ) {
    DispatchQueue.main.async { [weak self] in
      guard let self = self else { return }

      // 1. Immediately halt audio recording so no extraneous noise is appended
      if self.audioEngine.isRunning {
        self.audioEngine.stop()
      }
      if self.isTapInstalled {
        self.audioEngine.inputNode.removeTap(onBus: 0)
        self.isTapInstalled = false
      }

      // 2. Tell the speech recognition request that audio input has ended
      self.recognitionRequest?.endAudio()

      // 3. Give Apple's speech engine a brief moment (350ms) to transcribe the final words
      DispatchQueue.main.asyncAfter(deadline: .now() + 0.35) { [weak self] in
        guard let self = self else { return }
        let finalResult = self.currentTranscription.trimmingCharacters(in: .whitespacesAndNewlines)
        self.cleanupAudio()
        resolve(["text": finalResult])
      }
    }
  }

  // Cancel listening
  @objc(cancelListening:rejecter:)
  func cancelListening(
    _ resolve: @escaping RCTPromiseResolveBlock,
    rejecter reject: @escaping RCTPromiseRejectBlock
  ) {
    DispatchQueue.main.async { [weak self] in
      guard let self = self else { return }
      self.recognitionTask?.cancel()
      self.cleanupAudio()
      self.currentTranscription = ""
      resolve(["status": "cancelled"])
    }
  }

  // Check if speech recognition is available
  @objc(isAvailable:rejecter:)
  func isAvailable(
    _ resolve: @escaping RCTPromiseResolveBlock,
    rejecter reject: @escaping RCTPromiseRejectBlock
  ) {
    let recognizer = self.resolveSpeechRecognizer(preferredLocale: nil)
    let available = recognizer != nil
    let onDevice = (recognizer?.supportsOnDeviceRecognition ?? false)
    resolve([
      "available": available,
      "onDevice": onDevice,
      "locale": recognizer?.locale.identifier ?? "none"
    ])
  }

  // Text-To-Speech (TTS): Read text out loud completely offline
  @objc(speak:options:resolver:rejecter:)
  func speak(
    _ text: String,
    options: NSDictionary?,
    resolver resolve: @escaping RCTPromiseResolveBlock,
    rejecter reject: @escaping RCTPromiseRejectBlock
  ) {
    DispatchQueue.main.async { [weak self] in
      guard let self = self else { return }
      let clean = text.trimmingCharacters(in: .whitespacesAndNewlines)
      if clean.isEmpty {
        resolve(["status": "empty"])
        return
      }

      let queueMode = options?["queueMode"] as? String ?? "flush"
      if queueMode != "add" && self.speechSynthesizer.isSpeaking {
        self.speechSynthesizer.stopSpeaking(at: .immediate)
      }

      self.speechSynthesizer.delegate = self

      do {
        let audioSession = AVAudioSession.sharedInstance()
        try audioSession.setCategory(.playback, mode: .spokenAudio, options: [.duckOthers])
        try audioSession.setActive(true, options: .notifyOthersOnDeactivation)
      } catch {}

      let utterance = AVSpeechUtterance(string: clean)
      if let opts = options {
        if let rate = opts["rate"] as? NSNumber {
          utterance.rate = rate.floatValue * AVSpeechUtteranceDefaultSpeechRate
        } else {
          utterance.rate = AVSpeechUtteranceDefaultSpeechRate
        }
        if let pitch = opts["pitch"] as? NSNumber {
          utterance.pitchMultiplier = pitch.floatValue
        }
        if let lang = opts["language"] as? String, !lang.isEmpty {
          utterance.voice = AVSpeechSynthesisVoice(language: lang)
        } else {
          utterance.voice = AVSpeechSynthesisVoice(language: Locale.current.identifier) ?? AVSpeechSynthesisVoice(language: "en-US")
        }
      } else {
        utterance.rate = AVSpeechUtteranceDefaultSpeechRate
        utterance.voice = AVSpeechSynthesisVoice(language: Locale.current.identifier) ?? AVSpeechSynthesisVoice(language: "en-US")
      }

      self.speechSynthesizer.speak(utterance)
      resolve(["status": "speaking"])
    }
  }

  @objc(stopSpeaking:rejecter:)
  func stopSpeaking(
    _ resolve: @escaping RCTPromiseResolveBlock,
    rejecter reject: @escaping RCTPromiseRejectBlock
  ) {
    DispatchQueue.main.async { [weak self] in
      guard let self = self else { return }
      if self.speechSynthesizer.isSpeaking {
        self.speechSynthesizer.stopSpeaking(at: .immediate)
      }
      resolve(["status": "stopped"])
    }
  }

  @objc(isSpeaking:rejecter:)
  func isSpeaking(
    _ resolve: @escaping RCTPromiseResolveBlock,
    rejecter reject: @escaping RCTPromiseRejectBlock
  ) {
    resolve(["speaking": self.speechSynthesizer.isSpeaking])
  }

  private func cleanupAudio() {
    if audioEngine.isRunning {
      audioEngine.stop()
    }
    if isTapInstalled {
      audioEngine.inputNode.removeTap(onBus: 0)
      isTapInstalled = false
    }
    recognitionRequest = nil
    recognitionTask = nil
    try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
  }
}
