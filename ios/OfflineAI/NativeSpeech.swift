import Foundation
import AVFoundation
import Speech

@objc(NativeSpeech)
class NativeSpeech: NSObject, SFSpeechRecognizerDelegate {

  private var audioEngine = AVAudioEngine()
  private var speechRecognizer: SFSpeechRecognizer?
  private var recognitionRequest: SFSpeechAudioBufferRecognitionRequest?
  private var recognitionTask: SFSpeechRecognitionTask?
  private var currentTranscription: String = ""

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

      let locale = localeIdentifier.isEmpty ? Locale.current : Locale(identifier: localeIdentifier)
      self.speechRecognizer = SFSpeechRecognizer(locale: locale)

      guard let recognizer = self.speechRecognizer, recognizer.isAvailable else {
        reject("SPEECH_UNAVAILABLE", "Speech recognizer is not available for locale \(locale.identifier)", nil)
        return
      }

      recognizer.delegate = self
      self.currentTranscription = ""

      do {
        let audioSession = AVAudioSession.sharedInstance()
        try audioSession.setCategory(.playAndRecord, mode: .measurement, options: [.duckOthers, .defaultToSpeaker])
        try audioSession.setActive(true, options: .notifyOthersOnDeactivation)

        self.recognitionRequest = SFSpeechAudioBufferRecognitionRequest()
        guard let request = self.recognitionRequest else {
          reject("REQUEST_FAILED", "Unable to create speech recognition request", nil)
          return
        }

        // Prefer on-device offline recognition without internet
        if #available(iOS 13.0, *) {
          if recognizer.supportsOnDeviceRecognition {
            request.requiresOnDeviceRecognition = true
          }
        }

        request.shouldReportPartialResults = true

        let inputNode = self.audioEngine.inputNode
        let recordingFormat = inputNode.outputFormat(forBus: 0)

        inputNode.installTap(onBus: 0, bufferSize: 1024, format: recordingFormat) { buffer, _ in
          self.recognitionRequest?.append(buffer)
        }

        self.audioEngine.prepare()
        try self.audioEngine.start()

        self.recognitionTask = recognizer.recognitionTask(with: request) { result, error in
          if let result = result {
            self.currentTranscription = result.bestTranscription.formattedString
          }
          if error != nil || (result?.isFinal ?? false) {
            self.cleanupAudio()
          }
        }

        resolve(["status": "recording", "onDevice": request.requiresOnDeviceRecognition])
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

      let finalResult = self.currentTranscription
      self.recognitionRequest?.endAudio()
      self.cleanupAudio()

      resolve(["text": finalResult])
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
    let recognizer = SFSpeechRecognizer(locale: Locale.current)
    let available = recognizer?.isAvailable ?? false
    let onDevice = (recognizer?.supportsOnDeviceRecognition ?? false)
    resolve(["available": available, "onDevice": onDevice])
  }

  private func cleanupAudio() {
    if audioEngine.isRunning {
      audioEngine.stop()
      audioEngine.inputNode.removeTap(onBus: 0)
    }
    recognitionRequest = nil
    recognitionTask = nil
    try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
  }
}
