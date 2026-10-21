import Foundation
import UIKit
import Vision
import PDFKit

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

  @objc(extractPdfText:resolver:rejecter:)
  func extractPdfText(
    _ pdfUri: String,
    resolver resolve: @escaping RCTPromiseResolveBlock,
    rejecter reject: @escaping RCTPromiseRejectBlock
  ) {
    DispatchQueue.global(qos: .userInitiated).async {
      do {
        let fileUrl: URL
        if let parsedUrl = URL(string: pdfUri), parsedUrl.isFileURL {
          fileUrl = parsedUrl
        } else if pdfUri.hasPrefix("file://") {
          let unescaped = pdfUri.removingPercentEncoding ?? pdfUri
          fileUrl = URL(fileURLWithPath: String(unescaped.dropFirst(7)))
        } else {
          let unescaped = pdfUri.removingPercentEncoding ?? pdfUri
          fileUrl = URL(fileURLWithPath: unescaped)
        }

        guard let pdfDocument = PDFDocument(url: fileUrl) else {
          resolve("")
          return
        }

        var pagesText: [String] = []
        let totalPages = pdfDocument.pageCount
        let maxPages = min(totalPages, 50)

        for i in 0..<maxPages {
          guard let page = pdfDocument.page(at: i) else { continue }
          let pageText = page.string?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
          if !pageText.isEmpty {
            pagesText.append("[Page \(i + 1)]\n\(pageText)")
          } else {
            // Scanned PDF fallback via Vision OCR
            let pageRect = page.bounds(for: .mediaBox)
            if pageRect.width > 0 && pageRect.height > 0 {
              let renderer = UIGraphicsImageRenderer(size: pageRect.size)
              let image = renderer.image { ctx in
                UIColor.white.set()
                ctx.fill(pageRect)
                ctx.cgContext.translateBy(x: 0.0, y: pageRect.size.height)
                ctx.cgContext.scaleBy(x: 1.0, y: -1.0)
                page.draw(with: .mediaBox, to: ctx.cgContext)
              }
              if let cgImage = image.cgImage {
                let request = VNRecognizeTextRequest()
                request.recognitionLevel = .accurate
                let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
                try? handler.perform([request])
                if let observations = request.results as? [VNRecognizedTextObservation] {
                  let lines = observations.compactMap { $0.topCandidates(1).first?.string }
                  let ocrText = lines.joined(separator: "\n").trimmingCharacters(in: .whitespacesAndNewlines)
                  if !ocrText.isEmpty {
                    pagesText.append("[Page \(i + 1) (OCR)]\n\(ocrText)")
                  }
                }
              }
            }
          }
        }

        resolve(pagesText.joined(separator: "\n\n").trimmingCharacters(in: .whitespacesAndNewlines))
      } catch {
        resolve("")
      }
    }
  }
}
