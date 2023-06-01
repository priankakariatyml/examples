// Copyright 2019 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import CoreImage
import UIKit
import Accelerate
import MediaPipeTasksVision
import MediaPipeTasksText

/// A result from invoking the `Interpreter`.
struct Result {
  let inferenceTime: Double
  let inferences: [Inference]
}

/// An inference from invoking the `Interpreter`.
struct Inference {
  let confidence: Float
  let label: String
}

/// Information about a model file or labels file.
typealias FileInfo = (name: String, extension: String)

/// Information about the MobileNet model.
enum MobileNet {
  static let modelInfo: FileInfo = (name: "mobilenet_v2_1.0_224", extension: "tflite")
  static let labelsInfo: FileInfo = (name: "labels", extension: "txt")
}

enum CocoSsd {
  static let modelInfo: FileInfo = (name: "coco_ssd_mobilenet_v1_1.0_quant_2018_06_29", extension: "tflite")
  static let labelsInfo: FileInfo = (name: "labels", extension: "txt")
}

enum Bert {
  static let modelInfo: FileInfo = (name: "bert_text_classifier", extension: "tflite")
  static let labelsInfo: FileInfo = (name: "labels", extension: "txt")
}

enum Emb {
  static let modelInfo: FileInfo = (name: "mobilebert_embedding_with_metadata", extension: "tflite")
  static let labelsInfo: FileInfo = (name: "labels", extension: "txt")
}

enum FaceDetectionShortRange {
  static let modelInfo: FileInfo = (name: "face_detection_short_range", extension: "tflite")
  static let labelsInfo: FileInfo = (name: "labels", extension: "txt")
}

/// This class handles all data preprocessing and makes calls to run inference on a given frame
/// by invoking the `Interpreter`. It then formats the inferences obtained and returns the top N
/// results for a successful inference.
class ModelDataHandler: NSObject, ObjectDetectorLiveStreamDelegate, ImageClassifierLiveStreamDelegate, FaceDetectorLiveStreamDelegate {


  // MARK: - Internal Properties

  /// The current thread count used by the TensorFlow Lite Interpreter.
  var threadCount: Int

  let resultCount = 3
  let threadCountLimit = 10
  var imageClassifier: ImageClassifier!
  var objectDetector: ObjectDetector!
  var textClassifier: TextClassifier!
  var textEmbedder: TextEmbedder!
  
  var faceDetector: FaceDetector!

  // MARK: - Model Parameters

  let batchSize = 1
  let inputChannels = 3
  let inputWidth = 224
  let inputHeight = 224

  // MARK: - Private Properties

  /// List of labels from the given labels file.
  private var labels: [String] = []

  /// TensorFlow Lite `Interpreter` object for performing inference on a given model.
//  private var interpreter: Interpreter

  /// Information about the alpha component in RGBA data.
  private let alphaComponent = (baseOffset: 4, moduloRemainder: 3)

  // MARK: - Initialization
  

  /// A failable initializer for `ModelDataHandler`. A new instance is created if the model and
  /// labels files are successfully loaded from the app's main bundle. Default `threadCount` is 1.
  init?(modelFileInfo: FileInfo, labelsFileInfo: FileInfo, threadCount: Int = 1) {
    let modelFilename = modelFileInfo.name
    self.threadCount = 1
    super.init()
  

    // Construct the path to the model file.
    guard let modelPath = Bundle.main.path(
      forResource: modelFilename,
      ofType: modelFileInfo.extension
    ) else {
      print("Failed to load the model file with name: \(modelFilename).")
      return nil
    }
    
    guard let objModelPath = Bundle.main.path(
      forResource: CocoSsd.modelInfo.name,
      ofType: CocoSsd.modelInfo.extension
    ) else {
      print("Failed to load the model file with name: \(CocoSsd.modelInfo.name).")
      return nil
    }
    
    guard let embModelPath = Bundle.main.path(
      forResource: Emb.modelInfo.name,
      ofType: Emb.modelInfo.extension
    ) else {
      print("Failed to load the model file with name: \(CocoSsd.modelInfo.name).")
      return nil
    }
    
    guard let faceDetPath = Bundle.main.path(
      forResource: FaceDetectionShortRange.modelInfo.name,
      ofType: FaceDetectionShortRange.modelInfo.extension
    ) else {
      print("Failed to load the model file with name: \(FaceDetectionShortRange.modelInfo.name).")
      return nil
    }
    
    do {
            let objectDetectorOptions = ObjectDetectorOptions()
            objectDetectorOptions.baseOptions.modelAssetPath = objModelPath
            objectDetectorOptions.runningMode = RunningMode.liveStream
            objectDetectorOptions.objectDetectorLiveStreamDelegate = self
//
            objectDetector = try ObjectDetector(options: objectDetectorOptions)
//
      let imageClassifierOptions = ImageClassifierOptions()
      imageClassifierOptions.baseOptions.modelAssetPath = modelPath
      imageClassifierOptions.runningMode = RunningMode.liveStream
      imageClassifierOptions.imageClassifierLiveStreamDelegate = self
      imageClassifier = try ImageClassifier(options: imageClassifierOptions)
      
      let faceDetOptions = FaceDetectorOptions()
      faceDetOptions.baseOptions.modelAssetPath = faceDetPath
      faceDetOptions.runningMode = RunningMode.liveStream
      faceDetOptions.faceDetectorLiveStreamDelegate = self
      
      faceDetector = try FaceDetector(options: faceDetOptions)
//
//
      //    var imageClassifierResult: ImageClassifierResult
//      guard let modelPath = Bundle.main.path(
//        forResource: "bert_text_classifier",
//        ofType: "tflite"
//      ) else {
//        print("Failed to load the model")
//        return
//      }
      
//      do {
//        textClassifier = try TextClassifier(modelPath: modelPath)
//        textEmbedder = try TextEmbedder(modelPath: embModelPath)
//
//        imageClassifier = try ImageClassifier(options: imageClassifierOptions)
      
      
  

//        let textClassifierResult = try textClassifier?.classify(text: "unflinchingly bleak")
//        print(textClassifierResult?.classificationResult.classifications[0].categories[0].categoryName!)
//        //      imageClassifier = try ImageClassifier.classifier(options: options)
//        //        imageSearcher = try ImageSearcher.searcher(options: searcherOptions)
//      }
//      catch {
//        print("Failed to create the classifier with error: \(error.localizedDescription)")
//        return
//      }

//      DispatchQueue.main.asyncAfter(deadline: .now() + 10) {
//        self.imageClassifier = nil;
//        print("\(self.imageClassifier)");
//      }
//
      //      imageClassifier = try ImageClassifier.classifier(options: options)
      //        imageSearcher = try ImageSearcher.searcher(options: searcherOptions)
    }
    catch {
      print("Failed to create the classifier with error: \(error.localizedDescription)")
      return
    }

  }

  // MARK: - Internal Methods

  /// Performs image preprocessing, invokes the `Interpreter`, and processes the inference results.
  func runModel(onFrame sampleBuffer: CMSampleBuffer) -> Result? {
    
    guard let modelPath = Bundle.main.path(
      forResource: MobileNet.modelInfo.name,
      ofType: MobileNet.modelInfo.extension
    ) else {
      print("Failed to load the model file with name: \(MobileNet.modelInfo.name).")
      return nil
    }


    guard let objModelPath = Bundle.main.path(
      forResource: CocoSsd.modelInfo.name,
      ofType: CocoSsd.modelInfo.extension
    ) else {
      print("Failed to load the model file with name: \(CocoSsd.modelInfo.name).")
      return nil
    }

    guard let bertModelPath = Bundle.main.path(
      forResource: Bert.modelInfo.name,
      ofType: Bert.modelInfo.extension
    ) else {
      print("Failed to load the model file with name: \(CocoSsd.modelInfo.name).")
      return nil
    }

    guard let embModelPath = Bundle.main.path(
      forResource: Emb.modelInfo.name,
      ofType: Emb.modelInfo.extension
    ) else {
      print("Failed to load the model file with name: \(CocoSsd.modelInfo.name).")
      return nil
    }
    
    guard let faceDetPath = Bundle.main.path(
      forResource: FaceDetectionShortRange.modelInfo.name,
      ofType: FaceDetectionShortRange.modelInfo.extension
    ) else {
      print("Failed to load the model file with name: \(FaceDetectionShortRange.modelInfo.name).")
      return nil
    }

    do {

      let dur = CMTimeGetSeconds(CMSampleBufferGetPresentationTimeStamp(sampleBuffer));
      let durInMiliSec = Int(ceil(1000*dur));

      let image = try MPImage(sampleBuffer: sampleBuffer)

//
//      let faceDetOptions = FaceDetectorOptions()
//      faceDetOptions.baseOptions.modelAssetPath = faceDetPath
//      faceDetOptions.runningMode = RunningMode.liveStream
//      faceDetOptions.faceDetectorLiveStreamDelegate = self
//
//      let faceDetector = try FaceDetector(options: faceDetOptions)
//
      try faceDetector.detectAsync(image: image, timestampInMilliseconds: durInMiliSec)
//
//      print("\(faceDetector)");
//
//      let imageClassifierOptions = ImageClassifierOptions()
//      imageClassifierOptions.baseOptions.modelAssetPath = modelPath
//      imageClassifierOptions.runningMode = RunningMode.liveStream
//      imageClassifierOptions.imageClassifierLiveStreamDelegate = self
//
//      let imageClassifier = try ImageClassifier(options: imageClassifierOptions)

//      print("\(imageClassifier)");

//      let image = try MPImage(sampleBuffer: sampleBuffer)
//
////
//      let dur = CMTimeGetSeconds(CMSampleBufferGetPresentationTimeStamp(sampleBuffer));
//      let durInMiliSec = Int(ceil(1000*dur));
//
      try imageClassifier.classifyAsync(image: image, timestampInMilliseconds: durInMiliSec)
//
//
//      try imageClassifier.classifyAsync(image: image, timestampInMilliseconds: durInMiliSec)
//
//      let objectDetectorOptions = ObjectDetectorOptions()
//      objectDetectorOptions.baseOptions.modelAssetPath = objModelPath
//      objectDetectorOptions.runningMode = RunningMode.liveStream
//      objectDetectorOptions.objectDetectorLiveStreamDelegate = self
//
//      let objectDetector = try ObjectDetector(options: objectDetectorOptions)

//      print("\(objectDetector)");
      try objectDetector.detectAsync(image: image, timestampInMilliseconds: durInMiliSec)
//
////      let objectDetectionResult = try objectDetector?.detect(image: image)
////      print(objectDetectionResult?.detections[0].categories[0].categoryName)
////      print(objectDetectionResult?.detections[1].categories[0].categoryName)
////      print(objectDetectionResult?.detections[2].categories[0].categoryName)
////
//      let textClassifierOptions = TextClassifierOptions()
//      textClassifierOptions.baseOptions.modelAssetPath = bertModelPath
//
//      let textClassifier = try TextClassifier(options: textClassifierOptions)
//      let textClassifierResult = try textClassifier.classify(text: "unflinchingly bleak")
//      print(textClassifierResult.classificationResult.classifications[0].categories[0].categoryName!)
//
//      guard let embModelPath = Bundle.main.path(
//        forResource: Emb.modelInfo.name,
//        ofType: Emb.modelInfo.extension
//      ) else {
//        print("Failed to load the model file with name: \(CocoSsd.modelInfo.name).")
//        return nil
//      }
////
//      let textEmbedderOptions = TextEmbedderOptions()
//      textEmbedderOptions.baseOptions.modelAssetPath = embModelPath
//      let textEmbedder = try TextEmbedder(options: textEmbedderOptions)
//      let textEmbedderResult = try textEmbedder.embed(text: "unflinchingly bleak")
//      print(textEmbedderResult.embeddingResult.embeddings[0].quantizedEmbedding?[0])
//      print(textEmbedderResult.embeddingResult.embeddings[0].floatEmbedding?[0])

    }
    catch {
      print(error.localizedDescription)
    }
//

//     Return the inference time and inference results.
    return nil
  }
  
  
  func imageClassifier(_ imageClassifier: ImageClassifier, didFinishClassification imageClassifierResult: ImageClassifierResult?, timestampInMilliseconds: Int, error: Error?) {
    if(self.imageClassifier == imageClassifier) {
      print(imageClassifierResult!.classificationResult.classifications[0].categories[0].categoryName)
      print(imageClassifierResult!.classificationResult.classifications[0].categories[1].categoryName)
      print(imageClassifierResult!.classificationResult.classifications[0].categories[2].categoryName)
    }
  }
//
    func objectDetector(_ objectDetector: ObjectDetector, didFinishDetection objectDetectionResult: ObjectDetectorResult?, timestampInMilliseconds: Int, error: Error?) {
      if(self.objectDetector == objectDetector) {
        print(objectDetectionResult!.detections[0].categories[0].categoryName)
        print(objectDetectionResult!.detections[1].categories[0].categoryName)
        print(objectDetectionResult!.detections[2].categories[0].categoryName)
      }
    }
//
    func faceDetector(_ faceDetector: FaceDetector, didFinishDetection faceDetectorResult:FaceDetectorResult?, timestampInMilliseconds: Int, error: Error?) {

      if(self.faceDetector == faceDetector) {
        print(" Face Det called")
        
      }
    }
    
    
//    sleep(10)

  // MARK: - Private Methods

  /// Returns the top N inference results sorted in descending order.
  private func getTopN(results: [Float]) -> [Inference] {
    // Create a zipped array of tuples [(labelIndex: Int, confidence: Float)].
    let zippedResults = zip(labels.indices, results)

    // Sort the zipped results by confidence value in descending order.
    let sortedResults = zippedResults.sorted { $0.1 > $1.1 }.prefix(resultCount)

    // Return the `Inference` results.
    return sortedResults.map { result in Inference(confidence: result.1, label: labels[result.0]) }
  }

  /// Loads the labels from the labels file and stores them in the `labels` property.
  private func loadLabels(fileInfo: FileInfo) {
    let filename = fileInfo.name
    let fileExtension = fileInfo.extension
    guard let fileURL = Bundle.main.url(forResource: filename, withExtension: fileExtension) else {
      fatalError("Labels file not found in bundle. Please add a labels file with name " +
                   "\(filename).\(fileExtension) and try again.")
    }
    do {
      let contents = try String(contentsOf: fileURL, encoding: .utf8)
      labels = contents.components(separatedBy: .newlines)
    } catch {
      fatalError("Labels file named \(filename).\(fileExtension) cannot be read. Please add a " +
                   "valid labels file and try again.")
    }
  }

  /// Returns the RGB data representation of the given image buffer with the specified `byteCount`.
  ///
  /// - Parameters
  ///   - buffer: The pixel buffer to convert to RGB data.
  ///   - byteCount: The expected byte count for the RGB data calculated using the values that the
  ///       model was trained on: `batchSize * imageWidth * imageHeight * componentsCount`.
  ///   - isModelQuantized: Whether the model is quantized (i.e. fixed point values rather than
  ///       floating point values).
  /// - Returns: The RGB data representation of the image buffer or `nil` if the buffer could not be
  ///     converted.
  private func rgbDataFromBuffer(
    _ buffer: CVPixelBuffer,
    byteCount: Int,
    isModelQuantized: Bool
  ) -> Data? {
    CVPixelBufferLockBaseAddress(buffer, .readOnly)
    defer {
      CVPixelBufferUnlockBaseAddress(buffer, .readOnly)
    }
    guard let sourceData = CVPixelBufferGetBaseAddress(buffer) else {
      return nil
    }
    
    let width = CVPixelBufferGetWidth(buffer)
    let height = CVPixelBufferGetHeight(buffer)
    let sourceBytesPerRow = CVPixelBufferGetBytesPerRow(buffer)
    let destinationChannelCount = 3
    let destinationBytesPerRow = destinationChannelCount * width
    
    var sourceBuffer = vImage_Buffer(data: sourceData,
                                     height: vImagePixelCount(height),
                                     width: vImagePixelCount(width),
                                     rowBytes: sourceBytesPerRow)
    
    guard let destinationData = malloc(height * destinationBytesPerRow) else {
      print("Error: out of memory")
      return nil
    }
    
    defer {
        free(destinationData)
    }

    var destinationBuffer = vImage_Buffer(data: destinationData,
                                          height: vImagePixelCount(height),
                                          width: vImagePixelCount(width),
                                          rowBytes: destinationBytesPerRow)

    let pixelBufferFormat = CVPixelBufferGetPixelFormatType(buffer)

    switch (pixelBufferFormat) {
    case kCVPixelFormatType_32BGRA:
        vImageConvert_BGRA8888toRGB888(&sourceBuffer, &destinationBuffer, UInt32(kvImageNoFlags))
    case kCVPixelFormatType_32ARGB:
        vImageConvert_ARGB8888toRGB888(&sourceBuffer, &destinationBuffer, UInt32(kvImageNoFlags))
    case kCVPixelFormatType_32RGBA:
        vImageConvert_RGBA8888toRGB888(&sourceBuffer, &destinationBuffer, UInt32(kvImageNoFlags))
    default:
        // Unknown pixel format.
        return nil
    }

    let byteData = Data(bytes: destinationBuffer.data, count: destinationBuffer.rowBytes * height)
    if isModelQuantized {
        return byteData
    }

    // Not quantized, convert to floats
    let bytes = Array<UInt8>(unsafeData: byteData)!
    var floats = [Float]()
    for i in 0..<bytes.count {
        floats.append(Float(bytes[i]) / 255.0)
    }
    return Data(copyingBufferOf: floats)
  }
}

// MARK: - Extensions

extension Data {
  /// Creates a new buffer by copying the buffer pointer of the given array.
  ///
  /// - Warning: The given array's element type `T` must be trivial in that it can be copied bit
  ///     for bit with no indirection or reference-counting operations; otherwise, reinterpreting
  ///     data from the resulting buffer has undefined behavior.
  /// - Parameter array: An array with elements of type `T`.
  init<T>(copyingBufferOf array: [T]) {
    self = array.withUnsafeBufferPointer(Data.init)
  }
}

extension Array {
  /// Creates a new array from the bytes of the given unsafe data.
  ///
  /// - Warning: The array's `Element` type must be trivial in that it can be copied bit for bit
  ///     with no indirection or reference-counting operations; otherwise, copying the raw bytes in
  ///     the `unsafeData`'s buffer to a new array returns an unsafe copy.
  /// - Note: Returns `nil` if `unsafeData.count` is not a multiple of
  ///     `MemoryLayout<Element>.stride`.
  /// - Parameter unsafeData: The data containing the bytes to turn into an array.
  init?(unsafeData: Data) {
    guard unsafeData.count % MemoryLayout<Element>.stride == 0 else { return nil }
    #if swift(>=5.0)
    self = unsafeData.withUnsafeBytes { .init($0.bindMemory(to: Element.self)) }
    #else
    self = unsafeData.withUnsafeBytes {
      .init(UnsafeBufferPointer<Element>(
        start: $0,
        count: unsafeData.count / MemoryLayout<Element>.stride
      ))
    }
    #endif  // swift(>=5.0)
  }
}
