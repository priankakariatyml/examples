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
import TensorFlowLiteTaskVision
import TensorFlowLiteTaskText


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

/// This class handles all data preprocessing and makes calls to run inference on a given frame
/// by invoking the `Interpreter`. It then formats the inferences obtained and returns the top N
/// results for a successful inference.
class ModelDataHandler {

  // MARK: - Internal Properties

  /// The current thread count used by the TensorFlow Lite Interpreter.
  let threadCount: Int = 1

  let resultCount = 3
  let threadCountLimit = 10

  // MARK: - Model Parameters

  let batchSize = 1
  let inputChannels = 3
  let inputWidth = 224
  let inputHeight = 224

  // MARK: - Private Properties

  /// List of labels from the given labels file.
  private var labels: [String] = []

  /// TensorFlow Lite `Interpreter` object for performing inference on a given model.
  private var imageClassifier: ImageClassifier

  /// Information about the alpha component in RGBA data.
  private let alphaComponent = (baseOffset: 4, moduloRemainder: 3)

  // MARK: - Initialization

  /// A failable initializer for `ModelDataHandler`. A new instance is created if the model and
  /// labels files are successfully loaded from the app's main bundle. Default `threadCount` is 1.
  init?(modelFileInfo: FileInfo, labelsFileInfo: FileInfo, threadCount: Int = 1) {
    let modelFilename = modelFileInfo.name

    // Construct the path to the model file.
    guard let modelPath = Bundle.main.path(
      forResource: modelFilename,
      ofType: modelFileInfo.extension
    ) else {
      print("Failed to load the model file with name: \(modelFilename).")
      return nil
    }
    
    let options = ImageClassifierOptions(modelPath: modelPath)
    options.classificationOptions.maxResults = resultCount
  
    do {
        imageClassifier = try ImageClassifier.classifier(options: options)
      }
      catch {
        print("Failed to create the classifier with error: \(error.localizedDescription)")
        return nil
      }
    
    guard let textModelPath = Bundle.main.path(
      forResource: "test_model_nl_classifier_with_regex_tokenizer",
      ofType: modelFileInfo.extension
    ) else {
      print("Failed to load the model file with name: \(modelFilename).")
      return nil
    }
    
    
    let nLClassifier = TFLNLClassifier.nlClassifier(modelPath: textModelPath, options: TFLNLClassifierOptions())
  
    
    let categories = nLClassifier.classify(text: "This is the best movie I’ve seen in recent years.")
    print(categories)

    
  }

  // MARK: - Internal Methods

  /// Performs image preprocessing, invokes the `Interpreter`, and processes the inference results.
  func runModel(on sampleBuffer: CMSampleBuffer) -> Result? {
    
    do {
      guard let gmlImage = MLImage(sampleBuffer: sampleBuffer) else {
        return nil
      }
      
      let classificationResult = try imageClassifier.classify(
        mlImage: gmlImage)
            
      print("Top \(resultCount) Results")
      
      for (idx, category) in classificationResult.classifications[0].categories.enumerated() {
        guard let label = category.label else {
          return nil;
        }
        print("\(idx + 1): Label: \(label), Score: \(category.score)")
      }

    } catch let error {
      print("Failed to invoke the interpreter with error: \(error.localizedDescription)")
      return nil
    }

    return nil
  }

}
