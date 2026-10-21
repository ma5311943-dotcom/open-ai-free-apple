import Foundation
import UIKit
import Vision

@objc(NativeOCR)
class NativeOCR: NSObject {

  @objc
  static func requiresMainQueueSetup() -> Bool {
    return false
  }

  @objc(recognizeText:resolver:rejecter:)
  func recognizeText(
    _ imageUri: String,
    resolver resolve: @escaping RCTPromiseResolveBlock,
    rejecter reject: @escaping RCTPromiseRejectBlock
  ) {
    DispatchQueue.global(qos: .userInitiated).async {
      do {
        // Robust file URL decoding for iOS camera & gallery URIs
        let fileUrl: URL
        if let parsedUrl = URL(string: imageUri), parsedUrl.isFileURL {
          fileUrl = parsedUrl
        } else if imageUri.hasPrefix("file://") {
          let unescaped = imageUri.removingPercentEncoding ?? imageUri
          fileUrl = URL(fileURLWithPath: String(unescaped.dropFirst(7)))
        } else {
          let unescaped = imageUri.removingPercentEncoding ?? imageUri
          fileUrl = URL(fileURLWithPath: unescaped)
        }

        guard let data = try? Data(contentsOf: fileUrl) else {
          resolve("")
          return
        }

        let request = VNRecognizeTextRequest { (request, error) in
          if let error = error {
            reject("OCR_ERROR", error.localizedDescription, error)
            return
          }
          guard let observations = request.results as? [VNRecognizedTextObservation] else {
            resolve("")
            return
          }

          let recognizedLines = observations.compactMap { observation in
            observation.topCandidates(1).first?.string
          }
          let resultText = recognizedLines.joined(separator: "\n").trimmingCharacters(in: .whitespacesAndNewlines)
          resolve(resultText)
        }

        request.recognitionLevel = .accurate
        request.usesLanguageCorrection = true

        // VNImageRequestHandler(data:) automatically handles EXIF camera orientation and all iOS formats (HEIC, JPG, PNG)
        let handler = VNImageRequestHandler(data: data, options: [:])
        try handler.perform([request])
      } catch {
        resolve("")
      }
    }
  }
}
